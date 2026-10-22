"""Signing error handling in isolated fixtures; never build or access a real keychain."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class SigningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        project = Path(__file__).resolve().parents[2]
        self.signing_dir = self.root / "application-support-signing"
        self.scripts = self.root / "scripts"
        self.scripts.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "dirname").symlink_to("/usr/bin/dirname")
        source = (project / "scripts/signing.sh").read_text()
        source = source.replace('DIR="$HOME/Library/Application Support/Image Studio Signing"',
                                f'DIR="{self.signing_dir}"')
        self.helper = self.scripts / "signing.sh"
        self.helper.write_text(source)
        self.helper.chmod(0o755)
        # Exercise only the signing decision block, never build.sh or its build commands.
        build = (project / "build.sh").read_text()
        block = "if SIGNING=" + build.split("if SIGNING=", 1)[1].split('echo "Built $APP"', 1)[0]
        self.decision = self.root / "signing-decision.sh"
        self.decision.write_text('set -euo pipefail\nAPP="fixture.app"\n' + block)
        self.env = {**os.environ, "PATH":str(self.bin)}
        self.calls = self.root / "calls"
        self.stub("codesign", f'printf "%s\\n" "$*" >> "{self.calls}"\n')

    def stub(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def run_script(self, script):
        return subprocess.run(["/bin/bash", str(script)], cwd=self.root, env=self.env,
                              capture_output=True, text=True, timeout=5)

    def test_conflict_aborts_and_preserves_both_identities(self):
        old = self.root / ".signing"
        old.mkdir(); self.signing_dir.mkdir()
        (old / "identity").write_text("old")
        (self.signing_dir / "identity").write_text("new")
        for name in ("security", "openssl"):
            self.stub(name, f'printf "unexpected" >> "{self.calls}"\nexit 1\n')
        result = self.run_script(self.decision)
        self.assertEqual(result.returncode, 1)
        self.assertIn("Both .signing", result.stderr)
        self.assertIn("build aborted", result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertEqual((old / "identity").read_text(), "old")
        self.assertEqual((self.signing_dir / "identity").read_text(), "new")

    def test_only_missing_tools_allow_adhoc_fallback(self):
        old = self.root / ".signing"
        old.mkdir(); (old / "identity").write_text("old")
        for missing in ("openssl", "security", "both"):
            with self.subTest(missing=missing):
                for name in ("openssl", "security"):
                    (self.bin / name).unlink(missing_ok=True)
                    if missing not in (name, "both"):
                        self.stub(name, "exit 1\n")
                self.calls.unlink(missing_ok=True)
                result = self.run_script(self.decision)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("tools unavailable", result.stderr)
                self.assertEqual(self.calls.read_text().strip(), "--force --sign - fixture.app")
                self.assertFalse(self.signing_dir.exists())
                self.assertEqual((old / "identity").read_text(), "old")

    def test_other_setup_errors_abort_without_fallback(self):
        self.helper.write_text('#!/bin/bash\necho "fixture setup failure" >&2\nexit 42\n')
        result = self.run_script(self.decision)
        self.assertEqual(result.returncode, 42)
        self.assertIn("fixture setup failure", result.stderr)
        self.assertIn("build aborted", result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == "__main__":
    unittest.main()
