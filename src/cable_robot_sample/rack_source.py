import hashlib
import json
import os
from pathlib import Path
import tempfile


def configure_rack(directory, source):
    from pxr import Sdf, Tf, Usd

    root = Path(directory).expanduser().resolve(strict=True)
    source = Path(source).expanduser().resolve(strict=True)
    metadata = json.loads((root / 'rack_source.json').read_text(encoding='utf-8'))
    checksum = hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    if checksum.hexdigest() != metadata['sha256']:
        raise ValueError('Rack source SHA256 differs from the version used to prepare this scene')
    destination = root / 'external' / 'gb300.usda'
    if not destination.resolve().is_relative_to(root) or destination.resolve() == source:
        raise ValueError('Rack reference must stay inside the prepared scene and cannot replace the source')
    if destination.exists():
        try:
            existing = Sdf.Layer.FindOrOpen(str(destination))
        except Tf.ErrorException as error:
            raise FileExistsError(f'Refusing to replace an unreadable layer: {destination}') from error
        if existing is None or existing.customLayerData.get('cableSampleExternalRack') is not True:
            raise FileExistsError(f'Refusing to replace an unrecognized layer: {destination}')
    stage = Usd.Stage.CreateInMemory('rack.usda')
    layer = stage.GetRootLayer()
    layer.subLayerPaths = [source.as_posix()]
    layer.customLayerData = {'cableSampleExternalRack': True}
    destination.parent.mkdir(exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(suffix='.usda', dir=destination.parent)
    os.close(descriptor)
    try:
        if not layer.Export(temporary):
            raise RuntimeError('Could not write rack reference layer')
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'reference': str(destination), 'source': str(source), 'sha256': checksum.hexdigest()}
