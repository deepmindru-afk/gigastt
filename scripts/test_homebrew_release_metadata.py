"""Data-only release handoff fixtures; no API calls or publications."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('resolve-homebrew-release.py')
spec = importlib.util.spec_from_file_location('handoff', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleaseMetadataTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {'schema': 1, 'tag': 'v2.22.0', 'commit': 'a' * 40,
                         'repository': 'example/repo', 'run_id': 123, 'run_attempt': 2}
        self.event = {'workflow_run': {'id': 123, 'run_attempt': 2, 'name': 'Release',
                      'path': '.github/workflows/release.yml', 'event': 'workflow_dispatch',
                      'conclusion': 'success', 'head_branch': 'main',
                      'head_repository': {'full_name': 'example/repo'}}}

    def test_manual_main_and_tag_push_resolve_identical_validated_tag(self):
        self.assertEqual(module.resolve(self.metadata, self.event, 'example/repo'), 'v2.22.0')
        self.event['workflow_run'].update(event='push', head_branch='v2.22.0')
        self.assertEqual(module.resolve(self.metadata, self.event, 'example/repo'), 'v2.22.0')

    def test_failed_unrelated_or_foreign_run_is_rejected(self):
        for key, value in [('conclusion', 'failure'), ('name', 'CI'), ('path', '.github/workflows/other.yml'),
                           ('event', 'pull_request'), ('head_repository', {'full_name': 'other/repo'})]:
            event = copy.deepcopy(self.event)
            event['workflow_run'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                module.resolve(self.metadata, event, 'example/repo')

    def test_missing_invalid_or_wrong_run_metadata_is_rejected(self):
        for key, value in [('schema', 2), ('tag', 'main'), ('tag', 'v2.22.0\ninjected=x'),
                           ('commit', 'not-a-sha'), ('repository', 'other/repo'),
                           ('run_id', 124), ('run_attempt', 1), ('run_attempt', True)]:
            metadata = dict(self.metadata, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                module.resolve(metadata, self.event, 'example/repo')
        for key in self.metadata:
            metadata = self.metadata.copy()
            del metadata[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                module.resolve(metadata, self.event, 'example/repo')

    def test_cli_handoff_writes_only_validated_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'event.json').write_text(json.dumps(self.event))
            (root / 'metadata.json').write_text(json.dumps(self.metadata))
            output = root / 'output'
            result = subprocess.run([sys.executable, str(SCRIPT), '--metadata', str(root / 'metadata.json'),
                '--event', str(root / 'event.json'), '--repository', 'example/repo', '--output', str(output)], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(output.read_text(), 'tag=v2.22.0\n')

    def test_failed_cli_emits_no_tag_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'event.json').write_text(json.dumps(self.event))
            (root / 'metadata.json').write_text('{invalid json')
            output = root / 'output'
            result = subprocess.run([sys.executable, str(SCRIPT), '--metadata', str(root / 'metadata.json'),
                '--event', str(root / 'event.json'), '--repository', 'example/repo', '--output', str(output)], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())

    def test_manual_homebrew_dispatch_validates_tag_without_upstream_artifact(self):
        self.assertEqual(module.validate_tag('v2.22.0'), 'v2.22.0')
        for tag in ['main', 'v02.22.0', 'v2.22.0/evil', 'v2.22.0\ninjected=x']:
            with self.subTest(tag=tag), self.assertRaises(ValueError):
                module.validate_tag(tag)


if __name__ == '__main__':
    unittest.main()
