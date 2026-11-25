import hashlib
import json
from pathlib import Path

import numpy as np

from .admission import validate_admission


STATE_SHAPES = {
    'body_q': (24, 7), 'body_qd': (24, 6), 'body_f': (24, 6),
    'joint_q': (41,), 'joint_qd': (41,), 'targets': (9,),
}


class HeldStateBank:
    """Load the initial-state bank and validate its split and tensor integrity.

    Checks stored evidence and tensor integrity only. A runtime adapter must
    still verify body order, inertias, joint mapping and restored-state health.
    Bank-generation task metadata never overrides the selected PPO objective.
    """

    def __init__(self, directory, admission_path):
        directory = Path(directory)
        self.manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        admission = json.loads(Path(admission_path).read_text(encoding='utf-8'))
        archive_path = directory / 'states.npz'
        with archive_path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if self.manifest.get('phase') != 'accepted' or checksum != self.manifest.get('sha256'):
            raise ValueError('Initial-state bank not accepted or checksum mismatch')
        self.train_ids = tuple(validate_admission(self.manifest, admission))
        self.evaluation_ids = tuple(self.manifest['splits']['test'])
        count = sum(len(split) for split in (self.manifest['splits']['train'], self.evaluation_ids))
        self.states = {}
        with np.load(archive_path, allow_pickle=False) as archive:
            if not set(STATE_SHAPES).issubset(archive.files):
                raise ValueError('Missing initial-state arrays')
            for name, shape in STATE_SHAPES.items():
                value = archive[name]
                if value.shape != (count, *shape) or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError(f'Invalid initial-state array: {name}')
                value.flags.writeable = False
                self.states[name] = value

    def select(self, initial_ids, *, split):
        if split not in ('train', 'test'):
            raise ValueError('Split must be train or test')
        ids = np.asarray(initial_ids)
        if ids.ndim != 1 or ids.dtype.kind not in ('i', 'u'):
            raise ValueError('Initial IDs must be a one-dimensional integer array')
        allowed = self.train_ids if split == 'train' else self.evaluation_ids
        if not np.isin(ids, allowed).all():
            raise ValueError('Initial ID outside the admitted split')
        return {name: values[ids].copy() for name, values in self.states.items()}
