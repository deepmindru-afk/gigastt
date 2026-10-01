#!/usr/bin/env python3
"""Release upload validation without signing keys or network access."""
from pathlib import Path
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name('release-assets.py').resolve()


class ReleaseAssetsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.names = ['app.tar.gz', 'app.tar.gz.sha256', 'app.deb',
                      'SHA256SUMS.txt', 'app.cdx.json']
        for name in self.names:
            (self.root / name).write_text('fixture')

    def run_list(self, signed=False):
        return subprocess.run(['python3', str(SCRIPT), str(self.root),
                               '--signed' if signed else '--unsigned'],
                              text=True, capture_output=True)

    def sign_fixtures(self):
        for name in self.names:
            if not name.endswith('.sha256'):
                (self.root / (name + '.minisig')).write_text('signature fixture')

    def test_unsigned_includes_required_assets_only(self):
        (self.root / 'unrelated.minisig').write_text('stale')
        result = self.run_list()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()),
                         {str(self.root / name) for name in self.names})

    def test_signed_includes_each_expected_signature(self):
        self.sign_fixtures()
        result = self.run_list(signed=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(result.stdout.splitlines()),
                         {str(p) for p in self.root.iterdir()})

    def test_one_missing_signature_fails_without_partial_list(self):
        self.sign_fixtures()
        (self.root / 'app.deb.minisig').unlink()
        result = self.run_list(signed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')

    def test_every_mandatory_category_is_required(self):
        for name in self.names:
            with self.subTest(name=name):
                path = self.root / name
                path.unlink()
                result = self.run_list()
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
                path.write_text('fixture')


if __name__ == '__main__':
    unittest.main()
