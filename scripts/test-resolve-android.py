#!/usr/bin/env python3
"""Android source/version regressions using local Git repositories only."""
import importlib.util
import os
from pathlib import Path
import subprocess
import unittest

spec = importlib.util.spec_from_file_location('release_tests', Path(__file__).with_name('test-resolve-release.py'))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
SCRIPT = Path(__file__).with_name('resolve-android.py').resolve()


class AndroidSourceTests(unittest.TestCase):
    setUp = fixtures.ReleaseSourceTests.setUp
    git = fixtures.ReleaseSourceTests.git
    write_version = fixtures.ReleaseSourceTests.write_version

    def run_resolver(self, tag='', publish=False, commit=None):
        return subprocess.run(['python3', str(SCRIPT)], cwd=self.checkout,
                              env={**os.environ, 'RELEASE_TAG': tag,
                                   'ANDROID_PUBLISH': str(publish).lower(),
                                   'GITHUB_SHA': commit or self.dispatch,
                                   'GITHUB_OUTPUT': str(self.output)},
                              text=True, capture_output=True)

    def test_tag_overrides_dispatch_ref(self):
        result = self.run_resolver('v2.22.0', publish=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split('=', 1) for line in self.output.read_text().splitlines())
        self.assertEqual(outputs, {'tag': 'v2.22.0', 'commit': self.release, 'version': '2.22.0'})

    def test_untagged_build_uses_dispatch_commit(self):
        result = self.run_resolver()
        self.assertEqual(result.returncode, 0, result.stderr)
        outputs = dict(line.split('=', 1) for line in self.output.read_text().splitlines())
        self.assertEqual(outputs, {'tag': '', 'commit': self.dispatch, 'version': '2.23.0'})

    def test_untagged_publish_rejected_before_outputs(self):
        self.assertNotEqual(self.run_resolver(publish=True).returncode, 0)
        self.assertFalse(self.output.exists())

    def test_invalid_missing_or_mismatched_tag_rejected(self):
        self.git('tag', 'v2.24.0')
        for tag in ['main', 'v2.99.0', 'v2.24.0', 'v2.22.0\nversion=evil']:
            with self.subTest(tag=tag):
                self.assertNotEqual(self.run_resolver(tag).returncode, 0)
                self.assertFalse(self.output.exists())

    def test_mismatched_binding_version_rejected(self):
        self.write_version('2.24.0', member='2.3.0')
        self.git('tag', 'v2.24.0')
        self.assertNotEqual(self.run_resolver('v2.24.0').returncode, 0)
        self.assertFalse(self.output.exists())

    def test_untagged_source_requires_commit_sha(self):
        self.assertNotEqual(self.run_resolver(commit='HEAD').returncode, 0)
        self.assertFalse(self.output.exists())

    def test_workflow_pins_build_and_passes_validated_version(self):
        workflow = SCRIPT.parent.parent.joinpath('.github/workflows/android-aar.yml').read_text()
        self.assertIn('ref: ${{ needs.resolve.outputs.commit }}', workflow)
        self.assertIn('ANDROID_VERSION: ${{ needs.resolve.outputs.version }}', workflow)
        self.assertIn('gradle :gigastt:assembleRelease -PVERSION_NAME="$ANDROID_VERSION"', workflow)
        self.assertIn('gradle :gigastt:publish -PVERSION_NAME="$ANDROID_VERSION"', workflow)
        self.assertIn('gh release view "$RELEASE_TAG"', workflow)
        self.assertIn('gh release upload "$RELEASE_TAG"', workflow)
        self.assertNotIn('softprops/action-gh-release', workflow)


if __name__ == '__main__':
    unittest.main()
