import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from cable_robot_sample.recording.reader import Recording, cable_points, schedule


def main():
    parser = argparse.ArgumentParser(description='Render recorded cable states without advancing physics')
    parser.add_argument('session', type=Path)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--segment', type=int)
    parser.add_argument('--start', type=float, default=0.0)
    parser.add_argument('--end', type=float)
    parser.add_argument('--timebase', choices=('sim', 'wall'), default='sim')
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--camera', choices=('recorded', 'fixed'), default='recorded')
    parser.add_argument('--camera-eye', type=float, nargs=3)
    parser.add_argument('--camera-lookat', type=float, nargs=3)
    parser.add_argument('--hold-final', type=float, default=1.0)
    parser.add_argument('--no-forces', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--assets', type=Path, help='Relocated scene asset directory; defaults to recorded source')
    args = parser.parse_args()
    if (args.camera_eye is None) != (args.camera_lookat is None):
        parser.error('--camera-eye and --camera-lookat must be specified together')
    if args.camera_eye is not None:
        direction = np.asarray(args.camera_lookat) - np.asarray(args.camera_eye)
        if not np.isfinite([args.camera_eye, args.camera_lookat]).all() or np.linalg.norm(direction[:2]) < 1e-6:
            parser.error('Custom camera must be finite and have a nonvertical viewing direction')
    recording = Recording(args.session)
    if args.list:
        print(json.dumps(recording.summary(), indent=2))
        return
    if os.environ.get('OMNI_KIT_ACCEPT_EULA', '').upper() != 'YES':
        parser.error('Review and accept the installed Isaac Sim/Kit license yourself, then set OMNI_KIT_ACCEPT_EULA=YES. No rendering started.')
    segment, rows = recording.select_segment(args.segment)
    times, indices, stop = schedule(rows, args.timebase, args.start, args.end, args.fps, args.speed)
    if not np.isfinite(args.hold_final) or args.hold_final < 0:
        parser.error('--hold-final must be finite and nonnegative')
    if args.output is None:
        parser.error('Specify a new --output file.mp4 (or use --list)')
    destination = args.output.resolve()
    if destination.suffix.lower() != '.mp4':
        parser.error('--output must end in .mp4')
    partial = destination.with_suffix('.partial.mp4')
    report_path = destination.with_suffix('.json')
    log_path = destination.with_suffix('.ffmpeg.log')
    if any(path.exists() for path in (destination, partial, report_path, log_path)):
        raise FileExistsError('Output or evidence exists; choose a new output filename')
    encoder = shutil.which('ffmpeg')
    if encoder is None:
        raise RuntimeError('ffmpeg executable is required for offline video export')
    scene = recording.root / 'scene.usdc'
    if hashlib.sha256(scene.read_bytes()).hexdigest() != recording.metadata['scene_sha256']:
        raise ValueError('Scene snapshot hash differs from recording metadata')
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = dict(status='started', session=str(recording.session), segment=segment, start=args.start, end=stop,
        timebase=args.timebase, speed=args.speed, fps=args.fps, camera=args.camera, hold_final_seconds=args.hold_final,
        physical_steps_executed=0, force_readings='recorded CSV only, never recomputed',
        frames_written=0, output=str(destination))
    report.update(camera_eye=args.camera_eye, camera_lookat=args.camera_lookat)
    report_path.write_text(json.dumps(report, indent=2))
    app = None
    writer = None
    encoder_log = None
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=args.headless, enable_cameras=True, visualizer='kit', device='cuda:0')
        app = launcher.app
        import carb
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
        from cable_robot_sample.recording.overlay import ForceOverlay

        settings = carb.settings.get_settings()
        settings.set_bool('/app/player/playSimulations', False)
        settings.set_bool('/rtx/hydra/readTransformsFromFabricInRenderDelegate', False)
        rep.orchestrator.set_capture_on_play(False)
        omni.timeline.get_timeline_interface().stop()
        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageUpAxis(stage, 'Z')
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        source_stage = Usd.Stage.Open(str(scene))
        for spec in source_stage.GetRootLayer().rootPrims:
            if spec.name.startswith('Flattened_Prototype_'):
                Sdf.CopySpec(source_stage.GetRootLayer(), spec.path, stage.GetRootLayer(), spec.path)
        assets = (args.assets or Path(recording.metadata['source_assets'])).resolve(strict=True)
        for parent in ('/World', '/World/envs', '/World/envs/env_0'):
            stage.DefinePrim(parent, 'Xform')
        scene_assets = recording.metadata.get('scene_assets') or dict(
            Datacenter='rack.usda', **{curve['path'].split('/')[-1]: curve['path'].split('/')[-1] + '.usdc'
                                      for curve in recording.metadata['curves']})
        for name, relative in scene_assets.items():
            asset = (assets / relative).resolve(strict=True)
            if not asset.is_relative_to(assets) or not name.isidentifier():
                raise ValueError('Invalid recorded scene asset mapping')
            stage.DefinePrim('/World/envs/env_0/' + name, 'Xform').GetReferences().AddReference(str(asset))
        roots = [str(prim.GetPath()) for prim in source_stage.GetPrimAtPath('/World/envs/env_0').GetChildren()
                 if prim.GetName() not in scene_assets]
        for path in roots + ['/World/Ground', '/World/Light', '/World/CableAppearance']:
            if source_stage.GetPrimAtPath(path):
                Sdf.CopySpec(source_stage.GetRootLayer(), path, stage.GetRootLayer(), path)
        report['render_scene'] = 'Render-only reconstruction using recorded poses and user-supplied scene assets'
        for prim in list(stage.Traverse()):
            if 'Joint' in prim.GetTypeName():
                prim.SetActive(False)
                continue
            retained_apis = [name for name in prim.GetAppliedSchemas()
                if not any(token in name.lower() for token in ('physics', 'physx', 'newton', 'mjc'))]
            prim.SetMetadata('apiSchemas', Sdf.TokenListOp.CreateExplicit(retained_apis))
            for attribute in prim.GetAttributes():
                if attribute.GetName().split(':')[0].lower() in ('physics', 'physx', 'physxrigidbody', 'physxcollision', 'newton', 'mjc'):
                    prim.RemoveProperty(attribute.GetName())
        report['camera_intrinsics'] = dict(focal_length=24.0, horizontal_aperture=20.955,
            vertical_aperture=15.71625, clipping_range=[0.01, 100.0])
        report['source_chunks'] = [dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            for path in sorted({row[0] for row in rows})]
        operations = []
        rod_indices = {index for curve in recording.metadata['curves'] for index in curve['body_indices']}
        for body_index, label in enumerate(recording.metadata['body_labels']):
            if body_index in rod_indices:
                operations.append(None)
                continue
            prim = stage.GetPrimAtPath(label)
            if not prim:
                raise ValueError(f'Recorded body missing from scene: {label}')
            xform = UsdGeom.Xformable(prim)
            operation = xform.MakeMatrixXform()
            xform.SetResetXformStack(True)
            operations.append(operation)
        camera = UsdGeom.Camera.Define(stage, '/World/OfflineReplayCamera')
        camera_op = UsdGeom.Xformable(camera.GetPrim()).MakeMatrixXform()
        UsdGeom.Xformable(camera.GetPrim()).SetResetXformStack(True)
        products = rep.create.render_product(str(camera.GetPath()), (1280, 960))
        annotator = rep.AnnotatorRegistry.get_annotator('rgb', device='cpu')
        annotator.attach([products])
        cable_colors = []
        for curve in recording.metadata['curves']:
            if 'operable' in curve and not curve['operable']:
                continue
            if curve.get('connector_color'):
                linear_color = np.asarray(curve['connector_color']['color_linear'], dtype=float)
            else:
                surface = stage.GetPrimAtPath(curve['path'] + '/Material/Surface')
                linear_color = np.asarray(surface.GetAttribute('inputs:diffuseColor').Get(), dtype=float)
            if linear_color.shape != (3,) or not np.isfinite(linear_color).all():
                raise ValueError(f'Missing cable material color: {curve["path"]}')
            linear_color = np.clip(linear_color, 0.0, 1.0)
            srgb_color = np.where(linear_color <= 0.0031308, 12.92 * linear_color,
                1.055 * linear_color ** (1.0 / 2.4) - 0.055)
            cable_colors.append(tuple(np.rint(srgb_color * 255).astype(int).tolist()))
        overlay = None if args.no_forces or not (recording.session / 'forces/metadata.json').is_file() else ForceOverlay(recording, segment, rows, args.timebase, args.start, stop,
            speed=args.speed, cable_colors=cable_colors)
        report['cable_plot_colors_srgb'] = cable_colors
        if overlay is not None:
            report['recorded_performance'] = overlay.performance_summary
        width, height = (1280, 960) if overlay is None else (1920, 1080)
        encoder_log = log_path.open('xb')
        writer = subprocess.Popen([encoder, '-nostdin', '-n', '-loglevel', 'warning', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(args.fps), '-i', 'pipe:0',
            '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(partial)], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=encoder_log)
        scales = recording.metadata['body_world_scale']
        first_camera = None
        custom_camera_matrix = None if args.camera_eye is None else Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(*args.camera_eye), Gf.Vec3d(*args.camera_lookat), Gf.Vec3d(0, 0, 1)).GetInverse()
        last_image = None
        for number, (timestamp, frame_index) in enumerate(zip(times, indices)):
            if not app.is_running():
                raise RuntimeError('Renderer was closed before clip completed')
            frame = recording.load(rows[frame_index])
            poses = frame['body_q']
            if not bool(frame['finite']) or not np.isfinite(poses).all():
                raise ValueError('Clip includes a nonfinite state; preserve evidence and choose an earlier endpoint')
            with Sdf.ChangeBlock():
                for pose, scale, operation in zip(poses, scales, operations, strict=True):
                    if operation is None:
                        continue
                    transform = Gf.Transform()
                    transform.SetTranslation(Gf.Vec3d(*[float(value) for value in pose[:3]]))
                    transform.SetRotation(Gf.Rotation(Gf.Quatd(float(pose[6]), Gf.Vec3d(*[float(value) for value in pose[3:6]]))))
                    transform.SetScale(Gf.Vec3d(*scale))
                    operation.Set(transform.GetMatrix())
            for curve in recording.metadata['curves']:
                points = cable_points(poses, curve)
                primitive = UsdGeom.BasisCurves(stage.GetPrimAtPath(curve['path']))
                primitive.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                primitive.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.vstack([points.min(axis=0) - curve['radius_m'], points.max(axis=0) + curve['radius_m']])))
            if first_camera is None:
                first_camera = (frame['camera_matrix'].copy(), frame['camera_parameters'].copy(), bool(frame['camera_orthographic']))
            matrix, parameters, orthographic = first_camera if args.camera == 'fixed' else (frame['camera_matrix'], frame['camera_parameters'], bool(frame['camera_orthographic']))
            camera_op.Set(Gf.Matrix4d(*matrix.reshape(-1).tolist()) if custom_camera_matrix is None else custom_camera_matrix)
            camera.GetProjectionAttr().Set('orthographic' if orthographic else 'perspective')
            for attribute, value in zip((camera.GetFocalLengthAttr(), camera.GetHorizontalApertureAttr(), camera.GetVerticalApertureAttr(), camera.GetHorizontalApertureOffsetAttr(), camera.GetVerticalApertureOffsetAttr()), parameters[:5]):
                attribute.Set(float(value))
            camera.GetClippingRangeAttr().Set(Gf.Vec2f(float(parameters[5]), float(parameters[6])))
            if custom_camera_matrix is not None:
                camera.GetProjectionAttr().Set('perspective')
                camera.GetFocalLengthAttr().Set(24.0)
                camera.GetHorizontalApertureAttr().Set(20.955)
                camera.GetVerticalApertureAttr().Set(15.71625)
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
            settings.set_bool('/app/player/playSimulations', False)
            for _ in range(28 if number == 0 else 2):
                app.update()
            raw = annotator.get_data()
            if isinstance(raw, dict):
                raw = raw.get('data')
            rgb = np.asarray(raw, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[:2] != (960, 1280):
                raise RuntimeError(f'Unexpected rendered image shape: {rgb.shape}')
            rgb = rgb[:, :, :3]
            image = rgb if overlay is None else overlay.compose(rgb, float(timestamp), int(frame['control_step']), frame['kind'] != 1)
            if number in (0, len(times) // 2, len(times) - 1):
                from PIL import Image
                Image.fromarray(image).save(destination.with_name(f'{destination.stem}_frame{number:04d}.png'))
            last_image = np.ascontiguousarray(image).tobytes()
            writer.stdin.write(last_image)
            report['frames_written'] += 1
            if number % 120 == 0:
                print(f'RENDER {number + 1}/{len(times)} | {timestamp:.3f}s', flush=True)
        for _ in range(round(args.hold_final * args.fps)):
            writer.stdin.write(last_image)
            report['frames_written'] += 1
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError(f'ffmpeg failed; see {log_path}')
        os.replace(partial, destination)
        report.update(status='completed', video_seconds=report['frames_written'] / args.fps)
        print(json.dumps(report, indent=2))
    except Exception as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if writer is not None and writer.poll() is None:
            writer.terminate()
            try:
                writer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                writer.kill()
                writer.wait()
        if encoder_log is not None:
            encoder_log.close()
        report_path.write_text(json.dumps(report, indent=2))
        if app is not None:
            app.close(exit_code=0 if report['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
