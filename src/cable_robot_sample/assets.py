import json
import math
from pathlib import Path


def vector(values, length, label):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f'{label} must contain {length} numbers')
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError(f'{label} must contain finite numbers')


def validate_assets(directory, robot):
    root = Path(directory).expanduser().resolve(strict=True)
    robot = Path(robot).expanduser().resolve(strict=True)
    material_path = root / 'rod_material.json'
    if not material_path.is_file():
        material_path = Path(__file__).with_name('rod_material.json')
    required = [root / 'rack.usda', root / 'manifest.json', robot]
    if (root / 'rack_source.json').is_file() and not (root / 'external/gb300.usda').is_file():
        raise ValueError('External rack is not configured. Run configure-rack --assets <scene> --source <official external.usd> first')
    for path in required:
        if not path.is_file():
            raise ValueError(f'Missing asset file: {path}')
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    cables = manifest.get('cables')
    if not isinstance(cables, list) or not cables:
        raise ValueError('Manifest must have a nonempty cables list')
    for index, cable in enumerate(cables, 1):
        name = f'Cable{index}'
        if cable.get('name') != name:
            raise ValueError(f'Cables must be sequentially named Cable1..CableN; expected {name}')
        if not (root / f'{name}.usdc').is_file():
            raise ValueError(f'Missing {name}.usdc')
        count = cable.get('rod_count')
        if type(count) is not int or count < 2:
            raise ValueError(f'{name}: rod_count must be an integer >= 2')
        lengths = cable.get('segment_lengths_m')
        vector(lengths, count, f'{name}.segment_lengths_m')
        if min(lengths) <= 0:
            raise ValueError(f'{name}: segment lengths must be positive')
        vector(cable.get('jacket_color'), 3, f'{name}.jacket_color')
        if not all(0 <= value <= 1 for value in cable['jacket_color']):
            raise ValueError(f'{name}: jacket colors must be within [0, 1]')
        if 'connector_color' not in cable:
            raise ValueError(f'{name}: connector_color is required; use null for an uncolored connector')
        color = cable['connector_color']
        if color is not None:
            if not isinstance(color, dict):
                raise ValueError(f'{name}: connector_color must be an object or null')
            vector(color.get('color_linear'), 3, f'{name}.connector_color.color_linear')
            if not all(0 <= value <= 1 for value in color['color_linear']):
                raise ValueError(f'{name}: connector colors must be within [0, 1]')
        radius = cable.get('jacket_radius_m', 0)
        if not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
            raise ValueError(f'{name}: jacket_radius_m must be positive and finite')
        if type(cable.get('operable')) is not bool or not isinstance(cable.get('identifier'), str):
            raise ValueError(f'{name}: identifier and boolean operable are required')
        anchor = cable.get('bodies', {}).get('sc_plug_link', {})
        vector(anchor.get('position'), 3, f'{name}.anchor.position')
        quaternion = anchor.get('quaternion_xyzw')
        vector(quaternion, 4, f'{name}.anchor.quaternion_xyzw')
        if abs(sum(value * value for value in quaternion) - 1) > 1e-3:
            raise ValueError(f'{name}: anchor quaternion must be normalized')
    material = json.loads(material_path.read_text(encoding='utf-8'))
    for mode in ('stretch', 'shear', 'bend', 'twist'):
        units = ('N_per_m', 'N_s_per_m') if mode in ('stretch', 'shear') else ('Nm_per_rad', 'Nm_s_per_rad')
        for term, unit in zip(('stiffness', 'damping'), units):
            name = f'{mode}_{term}_{unit}'
            value = material.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'Material {name} must be finite and nonnegative')
    return {'cables': len(cables), 'rod_segments': sum(cable['rod_count'] for cable in cables),
            'operable': [cable['identifier'] for cable in cables if cable['operable']],
            'validation': 'manifest and files only; USD dependencies and physics not checked'}


def check_usd_dependencies(directory, robot):
    from pxr import UsdUtils

    root = Path(directory).resolve()
    sources = [root / 'rack.usda', Path(robot).resolve(), *sorted(root.glob('Cable*.usdc'))]
    unresolved = {}
    for source in sources:
        layers, assets, missing = UsdUtils.ComputeAllDependencies(str(source))
        if missing:
            unresolved[source.name] = [str(path) for path in missing]
    if unresolved:
        raise ValueError(f'Unresolved USD dependencies: {json.dumps(unresolved)}')
    return {'usd_sources': len(sources), 'unresolved': unresolved}
