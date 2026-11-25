import json
from pathlib import Path

import numpy as np


class Recording:
    def __init__(self, session):
        self.session = Path(session).resolve()
        self.root = self.session / 'trajectory'
        self.metadata = json.loads((self.root / 'metadata.json').read_text())
        if self.metadata['schema'] not in (1, 2):
            raise ValueError('Unsupported recording schema')
        self.frames = {}
        for line in (self.root / 'chunks.jsonl').read_text().splitlines(keepends=True):
            if not line.endswith('\n'):
                continue
            item = json.loads(line)
            path = (self.root / item['file']).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError('Invalid chunk path')
            with np.load(path, allow_pickle=False) as chunk:
                count = len(chunk['sim_time_s'])
                if count != item['frames']:
                    raise ValueError(f'Incomplete trajectory chunk: {path}')
                columns = {name: chunk[name] for name in ('sim_time_s', 'wall_time_s', 'control_step', 'kind')}
                rows = self.frames.setdefault(item['segment'], [])
                rows.extend((path, index, float(columns['sim_time_s'][index]), float(columns['wall_time_s'][index]),
                             int(columns['control_step'][index]), int(columns['kind'][index])) for index in range(count))
        self.cache_path = None
        self.cache = None

    def summary(self):
        return [dict(segment=segment, frames=len(rows), control_frames=sum(row[5] == 1 for row in rows),
                     sim_seconds=rows[-1][2], wall_seconds=rows[-1][3] - rows[0][3])
                for segment, rows in sorted(self.frames.items()) if rows]

    def select_segment(self, segment):
        if segment is None:
            active = [item['segment'] for item in self.summary() if item['control_frames']]
            if not active:
                raise ValueError('No recorded control steps; only paused/initial frames exist')
            segment = active[-1]
        if segment not in self.frames:
            raise ValueError(f'Unknown segment {segment}')
        return segment, self.frames[segment]

    def load(self, row):
        path, index = row[:2]
        if self.cache_path != path:
            with np.load(path, allow_pickle=False) as data:
                self.cache = {key: data[key] for key in data.files}
            self.cache_path = path
        return {key: value[index] for key, value in self.cache.items()}


def schedule(rows, timebase, start, end, fps, speed):
    if not rows or timebase not in ('sim', 'wall'):
        raise ValueError('Recording rows are required and timebase must be sim or wall')
    if not np.isfinite([start, fps, speed]).all() or fps <= 0 or speed <= 0 or start < 0:
        raise ValueError('Start, FPS and speed must be finite; FPS/speed positive and start nonnegative')
    column = 2 if timebase == 'sim' else 3
    origin = 0.0 if timebase == 'sim' else rows[0][3]
    times = np.array([row[column] - origin for row in rows])
    if np.any(np.diff(times) < -1e-9):
        raise ValueError('Recording timestamps are not monotonic')
    stop = float(times[-1]) if end is None else float(end)
    if not np.isfinite(stop) or stop <= start or stop > times[-1] + 1e-6 or start < times[0] - 1e-6:
        raise ValueError(f'Clip must be within [{times[0]:.4f}, {times[-1]:.4f}] seconds and have positive duration')
    samples = np.arange(start, stop, speed / fps)
    if not len(samples) or not np.isclose(samples[-1], stop):
        samples = np.append(samples, stop)
    indices = np.maximum(0, np.searchsorted(times, samples + 1e-8, side='right') - 1)
    return samples, indices, stop


def cable_points(poses, curve):
    from scipy.spatial.transform import Rotation

    selected = poses[curve['body_indices']]
    axes = Rotation.from_quat(selected[:, 3:]).apply(np.tile([1.0, 0.0, 0.0], (len(selected), 1)))
    first_half = curve.get('first_half_rod_length_m', curve.get('half_rod_length_m'))
    last_half = curve.get('last_half_rod_length_m', curve.get('half_rod_length_m'))
    start = selected[0, :3] - first_half * axes[0]
    end = selected[-1, :3] + last_half * axes[-1]
    return np.vstack([start, start, selected[:, :3], end, end]).astype(np.float32)
