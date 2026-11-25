import hashlib
import json
from pathlib import Path

import numpy as np

from .bank import STATE_SHAPES


def array_digest(values):
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def file_digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_member(directory, name):
    root = Path(directory).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Invalid reference member: {name}')
    return path


class HeldReference:
    """Load hash-checked rest coordinates and a held-state reference.

    Source directories are read-only. Only the rest-angle solver array is
    restored; archived warm solver caches are deliberately not replayed.
    """

    def __init__(self, directory, physical_directory):
        directory, physical_directory = Path(directory), Path(physical_directory)
        self.manifest = json.loads((directory / 'reference_manifest.json').read_text())
        expected_files = self.manifest['files_sha256']
        required = {'postgrasp_snapshot.npz', 'model_reference.npz', 'snapshot_metadata.json'}
        if not required.issubset(expected_files):
            raise ValueError('Incomplete reference manifest')
        for name, checksum in expected_files.items():
            if file_digest(checked_member(directory, name)) != checksum:
                raise ValueError(f'Reference checksum mismatch: {name}')
        schema = json.loads((directory / 'snapshot_metadata.json').read_text())
        self.body_labels = tuple(schema['body_labels'])
        self.joint_labels = tuple(self.manifest['joint_labels'])
        for labels in (self.body_labels, self.joint_labels):
            if len(labels) != len(set(labels)) or any('/env_0/' not in label for label in labels):
                raise ValueError('Reference labels must be unique and belong to env_0')
        status = json.loads((physical_directory / 'status.json').read_text())
        path = physical_directory / 'physical_reference.npz'
        if file_digest(path) != status['file_sha256']:
            raise ValueError('Physical reference checksum mismatch')
        with np.load(path, allow_pickle=False) as archive:
            self.structure = {}
            for name, checksum in self.manifest['structure_sha256'].items():
                value = archive[name]
                if array_digest(value) != checksum:
                    raise ValueError(f'Physical structure checksum mismatch: {name}')
                self.structure[name] = value.copy()
        with np.load(directory / 'postgrasp_snapshot.npz', allow_pickle=False) as archive:
            self.states = {name: archive['state/' + name].copy() for name in STATE_SHAPES if name != 'targets'}
            self.targets = archive['targets'].reshape(-1).copy()
            self.rest_angles = archive['solver/joint_rest_angle'].copy()
        with np.load(directory / 'model_reference.npz', allow_pickle=False) as archive:
            self.rest_poses = archive['body_q'].copy()
            self.rest_coordinates = archive['joint_q'].copy()
        values = dict(self.states, targets=self.targets, rest_angles=self.rest_angles,
                      rest_poses=self.rest_poses, rest_coordinates=self.rest_coordinates)
        shapes = dict(STATE_SHAPES, rest_angles=(41,), rest_poses=(24, 7), rest_coordinates=(41,))
        for name, value in values.items():
            if value.shape != shapes[name] or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError(f'Invalid reference array: {name}')
        if len(self.body_labels) != 24 or len(self.joint_labels) != len(self.structure['joint_type']):
            raise ValueError('Reference label count mismatch')


def map_worlds(reference, body_labels, joint_labels, arrays, num_envs, free_joint_type):
    """Validate mappings before mutating a Newton model; callable without Kit."""
    if len(set(body_labels)) != len(body_labels) or len(set(joint_labels)) != len(joint_labels):
        raise ValueError('Model labels are not unique')
    if np.any(arrays['joint_type'] == free_joint_type):
        raise ValueError('Free-joint translation is not supported by this fixed-base task')
    body_lookup = {label: index for index, label in enumerate(body_labels)}
    joint_lookup = {label: index for index, label in enumerate(joint_labels)}
    bodies, joints, coordinates, velocities, com_errors = [], [], [], [], []
    for world in range(num_envs):
        try:
            body_ids = np.array([body_lookup[label.replace('/env_0/', f'/env_{world}/')] for label in reference.body_labels])
            joint_ids = np.array([joint_lookup[label.replace('/env_0/', f'/env_{world}/')] for label in reference.joint_labels])
        except KeyError as error:
            raise ValueError(f'Reference body/joint absent in world {world}: {error}') from error
        if not np.all(arrays['body_world'][body_ids] == world):
            raise ValueError('Body mapping crosses simulation worlds')
        for name, indices in [('body_mass', body_ids), ('body_inertia', body_ids), ('joint_type', joint_ids)]:
            if array_digest(arrays[name][indices]) != reference.manifest['structure_sha256'][name]:
                raise ValueError(f'World {world} differs from reference: {name}')
        error = float(np.abs(arrays['body_com'][body_ids] - reference.structure['body_com']).max())
        if not np.isfinite(error) or error > 1e-7:
            raise ValueError(f'World {world} COM mismatch: {error}')
        com_errors.append(error)
        mappings = []
        for field, total in [('joint_q_start', arrays['joint_coord_count']), ('joint_qd_start', arrays['joint_dof_count'])]:
            starts = arrays[field]
            ids = [coordinate for joint in joint_ids for coordinate in range(
                int(starts[joint]), int(starts[joint + 1]) if joint + 1 < len(starts) else int(total))]
            if len(ids) != 41 or len(set(ids)) != 41:
                raise ValueError(f'World {world} coordinate mapping mismatch: {field}')
            mappings.append(ids)
        bodies.append(body_ids)
        joints.append(joint_ids)
        coordinates.append(mappings[0])
        velocities.append(mappings[1])
    return dict(bodies=np.asarray(bodies), joints=np.asarray(joints),
                coordinates=np.asarray(coordinates), velocities=np.asarray(velocities), com_errors=com_errors)
