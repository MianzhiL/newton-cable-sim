import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cable_robot_sample.learning.bank import HeldStateBank, STATE_SHAPES


class HeldBankTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        train, test = list(range(2049)), list(range(2049, 2334))
        self.arrays = {name: np.zeros((2334, *shape), dtype=np.float32) for name, shape in STATE_SHAPES.items()}
        self.manifest = dict(phase='accepted', splits=dict(train=train, test=test),
                             task_contract={'gamma': .999})
        self.admission = dict(
            mode='per_state_admission', infrastructure_passed=True,
            canonical_grip_unchanged=True, exact_retained_state_copy=True,
            mapped_parent_scan_count=2334, peer_reset_checks=3,
            train_test_separation=True, quarantined_train_ids=[0],
            active_train_ids=train[1:], evaluation_ids=test, maximum_reset_attempts=3,
        )
        self.write_bank()

    def write_bank(self):
        np.savez(self.root / 'states.npz', **self.arrays)
        checksum = hashlib.sha256((self.root / 'states.npz').read_bytes()).hexdigest()
        self.manifest['sha256'] = checksum
        self.admission['bank_sha256'] = checksum
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        (self.root / 'admission.json').write_text(json.dumps(self.admission))

    def load(self):
        return HeldStateBank(self.root, self.root / 'admission.json')

    def test_exact_split_and_owned_selection(self):
        bank = self.load()
        self.assertEqual(len(bank.train_ids), 2048)
        self.assertEqual(bank.evaluation_ids, tuple(range(2049, 2334)))
        selected = bank.select([1, 1, 2], split='train')
        selected['targets'][0, 0] = 7
        self.assertEqual(bank.states['targets'][1, 0], 0)
        self.assertEqual(selected['targets'][1, 0], 0)
        self.assertEqual(bank.select(bank.evaluation_ids, split='test')['body_q'].shape, (285, 24, 7))
        self.assertFalse(bank.states['body_q'].flags.writeable)

    def test_no_quarantine_leak_or_evaluation_replacement(self):
        bank = self.load()
        for ids, split in [([0], 'train'), ([1], 'test'), ([2049], 'train'),
                           ([-1], 'test'), ([1.0], 'train'), ([True], 'train'),
                           ([[1]], 'train'), ([1], 'other')]:
            with self.subTest(ids=ids, split=split), self.assertRaises(ValueError):
                bank.select(ids, split=split)

    def test_reject_corrupt_or_incompatible_bank(self):
        with (self.root / 'states.npz').open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaises(ValueError):
            self.load()
        for name, replacement in [('body_q', np.zeros((2334, 23, 7), np.float32)),
                                  ('targets', np.zeros((2334, 9), np.float64)),
                                  ('body_f', np.full((2334, 24, 6), np.nan, np.float32))]:
            original = self.arrays[name]
            self.arrays[name] = replacement
            self.write_bank()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.load()
            self.arrays[name] = original


if __name__ == '__main__':
    unittest.main()
