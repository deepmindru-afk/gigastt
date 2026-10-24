import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('objc_gate', Path(__file__).with_name('compare-objc-revisions.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    mode = {'seconds': 1.0, 'load_seconds': 0.5, 'cached': True,
            'outputs': [{'sha256': 'a' * 64, 'shape': [1, 1024, 192]} for _ in range(3)]}
    row = {'peak_rss_bytes': 1000, 'fixtures': ['b' * 64] * 3,
           'modes': {'cpu': mode, 'cpu_and_ne': copy.deepcopy(mode)}}
    return {arm: [copy.deepcopy(row) for _ in range(7)] for arm in ['base', 'candidate']}


class GateTests(unittest.TestCase):
    def test_identical_passes(self):
        module.validate(fixture())

    def test_missing_round_fails(self):
        r = fixture(); r['candidate'].pop()
        with self.assertRaises(ValueError): module.validate(r)

    def test_changed_output_fails(self):
        r = fixture(); r['candidate'][2]['modes']['cpu']['outputs'][0]['sha256'] = 'c' * 64
        with self.assertRaises(ValueError): module.validate(r)

    def test_invalid_metrics_fail(self):
        for value in [0, -1, float('nan'), float('inf')]:
            r = fixture(); r['candidate'][0]['modes']['cpu']['seconds'] = value
            with self.subTest(value=value), self.assertRaises(ValueError): module.validate(r)

    def test_slowdown_fails(self):
        r = fixture()
        for row in r['candidate']: row['modes']['cpu']['seconds'] = 1.1
        with self.assertRaises(ValueError): module.validate(r)

    def test_memory_regression_fails(self):
        r = fixture()
        for row in r['candidate']: row['peak_rss_bytes'] = 1101
        with self.assertRaises(ValueError): module.validate(r)

    def test_changed_assets_or_cold_cache_fails(self):
        r = fixture(); r['candidate'][0]['fixtures'][0] = 'c' * 64
        with self.assertRaises(ValueError): module.validate(r)
        r = fixture(); r['candidate'][0]['modes']['cpu']['cached'] = False
        with self.assertRaises(ValueError): module.validate(r)


class AffectedTests(unittest.TestCase):
    def changed(self, base_source, candidate_source, paths=''):
        def git(*args):
            if args[0] == 'show':
                source = base_source if args[1].startswith('base:') else candidate_source
                return '[[package]]\nname = "objc2"\nversion = "0.6.5"\nsource = "' + source + '"\n'
            return paths
        with mock.patch.object(module, 'git', side_effect=git):
            return module.affected('base', 'candidate')

    def test_unchanged_runtime_skips_measurement(self):
        self.assertFalse(self.changed('registry+same', 'registry+same'))

    def test_source_revision_change_is_detected(self):
        self.assertTrue(self.changed('git+repo#old', 'git+repo#new'))

    def test_bridge_change_is_detected(self):
        self.assertTrue(self.changed('registry+same', 'registry+same',
                                    'crates/gigastt-core/src/runtime/coreml/bridge/predict.rs'))

    def test_test_only_change_skips_measurement(self):
        self.assertFalse(self.changed('registry+same', 'registry+same',
                                     'crates/gigastt-core/src/runtime/coreml/bridge/tests/objc_lifetime.rs'))


if __name__ == '__main__': unittest.main()
