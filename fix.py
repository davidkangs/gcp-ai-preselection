#!/usr/bin/env python3
"""
patch_fix_tempdir.py
────────────────────────────────────────────────────────
● NamedTemporaryFile → mkdtemp + 수동 정리
● 들여쓰기 오류(IndentationError) 자동 복구
● 백업본: *.bak 생성
"""

import re, os, shutil, tempfile
from pathlib import Path

FILES = ["process3_inference.py", "process3_batch.py"]

# ────────── 교체할 코드 조각 ──────────
TEMPLATE = """\
{indent}# ── 임시 폴더 생성 ─────────────────────────
{indent}tmpdir_path = tempfile.mkdtemp()
{indent}temp_path   = os.path.join(tmpdir_path, "temp_road.shp")
{indent}{first_line_after}
"""

# 패턴:  with tempfile.NamedTemporaryFile( ... .shp ... ) as tmp:
WITH_PAT = re.compile(
    r"with\s+tempfile\.NamedTemporaryFile\([^\n]*?\.shp[^\n]*?\)\s+as\s+\w+\s*:\s*\n"
    r"\s*temp_path\s*=\s*\w+\.name\s*\n", re.MULTILINE
)

def patch_one(path: Path):
    txt = path.read_text(encoding="utf-8")

    # 1) import os, shutil, tempfile 확인
    header = txt.splitlines()[:40]
    if not any("import os" in l for l in header):
        txt = txt.replace("import tempfile", "import tempfile\nimport os", 1)
    if not any("import shutil" in l for l in header):
        txt = txt.replace("import os", "import os\nimport shutil", 1)

    # 2) WITH 블록 치환
    def _repl(match):
        # 들여쓰기 깊이 계산
        line = match.group(0).splitlines()[0]
        indent = re.match(r"\s*", line).group(0)
        # 블록 다음 첫 코드 줄을 찾아 그대로 유지
        after_pos = match.end()
        after_line = re.search(r"[^\s\n].*", txt[after_pos:]).group(0)
        return TEMPLATE.format(indent=indent, first_line_after=after_line)

    txt = WITH_PAT.sub(_repl, txt)

    # 3) 불필요한 Path(...).unlink(…) 삭제
    txt = re.sub(r"[ \t]*Path\(temp_path\)\.unlink\(.*?\)\n", "", txt)
    txt = re.sub(r"[ \t]*for ext in \[.*?unlink\(.*?\)\n", "", txt, flags=re.DOTALL)

    # 4) 콜백에서 임시폴더 삭제 함수 삽입(이미 있다면 생략)
    if "def cleanup_tmpdir(" not in txt:
        cleanup_func = """
def cleanup_tmpdir(worker):
    if hasattr(worker, 'tmpdir_path') and os.path.exists(worker.tmpdir_path):
        shutil.rmtree(worker.tmpdir_path, ignore_errors=True)
"""
        txt += cleanup_func

    # 5) on_prediction_completed / on_prediction_error 에 cleanup 호출
    txt = txt.replace(
        "def on_prediction_completed(",
        "def on_prediction_completed(",
    ).replace(
        "QMessageBox.warning(",
        "QMessageBox.warning(",
    )
    txt = txt.replace(
        ".prediction_completed.connect(",
        ".prediction_completed.connect(lambda r: (cleanup_tmpdir(self.prediction_worker), r and None) and None).connect(",
    ).replace(
        ".error_occurred.connect(",
        ".error_occurred.connect(lambda e: (cleanup_tmpdir(self.prediction_worker), e and None) and None).connect(",
    )

    # 6) 백업 & 저장
    bak = path.with_suffix(".bak")
    bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(txt, encoding="utf-8")
    print(f"✔ {path.name} patched  (backup → {bak.name})")

def main():
    root = Path(__file__).parent
    for fname in FILES:
        p = root / fname
        if p.exists():
            patch_one(p)
        else:
            print(f"⚠ {fname} not found – 건너뜀")

if __name__ == "__main__":
    main()
