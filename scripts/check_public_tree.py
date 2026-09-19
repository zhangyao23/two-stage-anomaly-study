"""Reject accidentally staged personal notes, raw data, weights and obvious credentials."""

import re
import subprocess
import sys
from pathlib import PurePosixPath


def check():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode("utf-8").split("\0")[:-1]
    issues = []
    for name in paths:
        path = PurePosixPath(name)
        private_name = (
            "闈㈣瘯" in name
            or "interview" in name.lower()
            or path.name in {"TASK_STATE.md", "CV_CANDIDATE.md"}
        )
        blocked_dir = any(
            p in {"local-only", "data", "checkpoints", ".venv", "__pycache__"} for p in path.parts
        )
        if private_name or blocked_dir or path.suffix in {".pt", ".pth", ".npz", ".docx", ".xlsx"}:
            issues.append(name + ": private/raw/generated path")
            continue
        content = subprocess.check_output(["git", "show", ":" + name])
        if len(content) > 2_000_000:
            issues.append(name + ": unexpectedly large file")
        if path.suffix == ".png":
            continue
        text = content.decode("utf-8")
        for pattern in [
            r"gh[pousr]_[A-Za-z0-9]{20,}",
            r"github_pat_[A-Za-z0-9_]{20,}",
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
            r"AKIA[0-9A-Z]{16}",
        ]:
            if re.search(pattern, text):
                issues.append(name + ": possible credential (value suppressed)")
        if path.suffix == ".md" and "For interview preparation" in text:
            issues.append(name + ": personal preparation link")
    if issues:
        print("Public-tree audit failed:\n" + "\n".join(issues))
        return 1
    print(f"Public-tree audit passed: {len(paths)} staged/tracked files.")
    return 0


if __name__ == "__main__":
    sys.exit(check())
