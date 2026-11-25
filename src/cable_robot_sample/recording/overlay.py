import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ForceOverlay:
    def __init__(self, recording, segment, rows, timebase, start, stop, speed=1.0, cable_colors=None):
        path = recording.session / 'forces/summary.csv'
        self.start, self.stop, self.timebase = start, stop, timebase
        self.segment = segment
        self.speed = speed
        curves = [curve for curve in recording.metadata['curves'] if curve.get('operable', True)]
        sensor_names = [curve['path'].split('/')[-1].lower() + '_fixed' for curve in curves]
        self.cable_labels = [curve.get('identifier', curve['path'].split('/')[-1]) for curve in curves]
        cable_count = len(curves)
        step_times = {row[4]: row[2] if timebase == 'sim' else row[3] - rows[0][3] for row in rows if row[5] == 1}
        self.samples = []
        if path.is_file():
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    step = int(row['control_step'])
                    if int(row['segment']) != segment or step not in step_times:
                        continue
                    values = [float(row[f'wrist_{component}_mean']) for component in ('Fx_N', 'Fy_N', 'Fz_N', 'Tx_Nm', 'Ty_Nm', 'Tz_Nm')]
                    values += [float(row[name + '_force_norm_max_N']) for name in sensor_names]
                    self.samples.append((step_times[step], int(row['valid']), values))
        self.times = np.array([sample[0] for sample in self.samples])
        self.values = np.array([sample[2] for sample in self.samples]).reshape(-1, 6 + cable_count)
        if len(self.samples):
            self.values[np.array([sample[1] == 0 for sample in self.samples])] = np.nan
        self.colors = [(235, 100, 92), (118, 185, 70), (84, 165, 245)]
        self.cable_colors = cable_colors
        if self.cable_colors is None or len(self.cable_colors) != cable_count:
            raise ValueError('One color per plotted cable is required')
        self.performance = []
        events_path = recording.session / 'events.jsonl'
        if events_path.is_file():
            for line in events_path.read_text().splitlines():
                event = json.loads(line)
                duration_ms = event.get('wall_step_ms', 0.0)
                if event.get('event') == 'step' and event['step'] in step_times and np.isfinite(duration_ms) and duration_ms > 0:
                    self.performance.append((event['step'], duration_ms))
        self.performance.sort()
        force_metadata = json.loads((recording.session / 'forces/metadata.json').read_text())
        self.substeps = force_metadata['physics_hz'] / force_metadata['control_hz']
        durations = [duration for step, duration in self.performance if start - 1e-8 <= step_times[step] <= stop + 1e-8]
        self.performance_summary = dict(source='events.jsonl wall_step_ms', sampling='one timed env.step every 60 control steps',
            includes='environment step, rendering, force capture and cable visual update',
            excludes='XR input/retargeting, trajectory capture and pauses', physics_substeps_per_env_step=self.substeps,
            samples=len(durations), sampled_env_steps_per_second=None if not durations else 1000.0 / float(np.mean(durations)))
        font = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
        self.font = ImageFont.truetype(str(font), 20) if font.exists() else ImageFont.load_default()
        self.small = ImageFont.truetype(str(font), 16) if font.exists() else self.font
        self.bounds = []
        selected = (self.times >= start - 1e-8) & (self.times <= stop + 1e-8)
        for offset in (0, 3, 6):
            values = self.values[selected, offset:offset + (cable_count if offset == 6 else 3)]
            finite = np.abs(values[np.isfinite(values)])
            self.bounds.append(max(0.01, float(finite.max()) if finite.size else 0.01) * 1.1)

    def compose(self, rgb, timestamp, control_step, paused):
        canvas = Image.new('RGB', (1920, 1080), (18, 22, 28))
        canvas.paste(Image.fromarray(rgb).resize((1280, 960)), (0, 60))
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), f'Optical-module cable teleoperation | Segment {self.segment} | {self.timebase} {timestamp:.2f} s | step {control_step}', font=self.font, fill='white')
        if self.speed != 1.0:
            playback = 'SLOW MOTION' if self.speed < 1.0 else 'PLAYBACK'
            draw.text((1320, 15), f'{playback} {self.speed:g}x | {self.timebase} time', font=self.font, fill=(235, 220, 140))
        past_samples = [entry for entry in self.performance if entry[0] <= control_step]
        rate_text = 'No timing sample yet'
        if past_samples:
            rate = 1000.0 / past_samples[-1][1]
            rate_text = f'{rate:.1f} env steps/s | {rate * self.substeps:.0f} physics substeps/s'
        draw.text((20, 1028), 'Original-run env.step (sampled every 60 steps): ' + rate_text,
            font=self.font, fill=(205, 210, 220))
        draw.text((20, 1056), f'1 env step = {self.substeps:g} physics substeps. Includes env rendering/force capture; excludes XR/trajectory recording. No physics re-simulation.',
            font=self.small, fill=(180, 185, 195))
        available = (self.times >= self.start) & (self.times <= timestamp + 1e-8)
        for panel, (offset, title, unit) in enumerate(((0, 'Wrist force - local frame, mean', 'N'), (3, 'Wrist torque - local frame, mean', 'N m'), (6, 'Fixed ends - force norm, substep peak', 'N'))):
            left, right, top, bottom = 1355, 1895, 120 + panel * 315, 300 + panel * 315
            bound = self.bounds[panel]
            draw.text((1300, top - 55), title, font=self.font, fill='white')
            colors = self.colors if panel < 2 else self.cable_colors
            labels = ('X', 'Y', 'Z') if panel < 2 else self.cable_labels
            draw.text((1300, top - 27), unit, font=self.small, fill=(200, 205, 215))
            for component, (label, color) in enumerate(zip(labels, colors)):
                legend_x = 1390 + component * (500 // len(labels))
                draw.line((legend_x, top - 17, legend_x + 20, top - 17), fill=color, width=3)
                draw.text((legend_x + 27, top - 27), label, font=self.small, fill=color)
            ymin = -bound if panel < 2 else 0.0
            for fraction in (0.0, 0.5, 1.0):
                ypos = top + fraction * (bottom - top)
                draw.line((left, ypos, right, ypos), fill=(58, 65, 76))
                draw.text((1290, ypos - 7), f'{bound - fraction * (bound - ymin):.2g}', font=self.small, fill=(190, 195, 205))
            draw.text((left, bottom + 10), f'{self.start:.1f}s', font=self.small, fill='white')
            draw.text((right - 65, bottom + 10), f'{self.stop:.1f}s', font=self.small, fill='white')
            indices = np.flatnonzero(available)
            if not len(indices):
                draw.text((left + 15, top + 60), 'No synchronized force samples', font=self.small, fill=(190, 195, 205))
                continue
            indices = indices[np.unique(np.linspace(0, len(indices) - 1, min(len(indices), 750)).astype(int))]
            for component, color in enumerate(colors):
                line = []
                for index in indices:
                    value = self.values[index, offset + component]
                    if not np.isfinite(value):
                        if len(line) > 1:
                            draw.line(line, fill=color, width=2)
                        line = []
                        continue
                    xpos = left + (self.times[index] - self.start) / (self.stop - self.start) * (right - left)
                    ypos = bottom - (value - ymin) / (bound - ymin) * (bottom - top)
                    line.append((xpos, ypos))
                if len(line) > 1:
                    draw.line(line, fill=color, width=2)
        return np.asarray(canvas)
