"""把每篇课程当作脚本完整跑一遍（练习单元用参考答案替换），用来保证所有代码真的能跑通。

用法：
  python tools/run_lessons.py                    # 跑全部课程
  python tools/run_lessons.py 01_prerequisites   # 只跑名字里包含该字符串的课程
  python tools/run_lessons.py --keep             # 保留生成的脚本，便于调试
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import LESSONS, ROOT, is_index, parse_lesson  # noqa: E402


def lesson_to_script(path: Path) -> str:
    cells = parse_lesson(path.read_text(encoding="utf-8"))
    chunks = [f"import sys; sys.path.insert(0, {str(ROOT)!r})", "import matplotlib; matplotlib.use('Agg')"]
    for k, c in enumerate(cells):
        if c.kind == "code":
            # 练习单元也执行一遍（它可能定义了后面要用的辅助函数），但忽略其中 TODO 引发的异常，
            # 随后参考答案会覆盖同名函数
            nxt = cells[k + 1] if k + 1 < len(cells) else None
            if nxt is not None and nxt.kind == "solution":
                chunks.append(
                    f"# ==== exercise {k} ====\ntry:\n    exec(compile({c.source!r}, 'exercise_{k}', 'exec'), globals())\n"
                    "except NotImplementedError:\n    pass"
                )
                continue
            chunks.append(f"# ==== cell {k} ====\n" + c.source)
        elif c.kind == "solution":
            chunks.append(f"# ==== solution {k} ====\n" + c.source)
    return "\n\n".join(chunks) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("-v", "--verbose", action="store_true", help="通过的课程也打印输出")
    args = ap.parse_args()

    files = [p for p in sorted(LESSONS.rglob("*.md")) if not is_index(p)]
    if args.filters:
        files = [p for p in files if any(f in str(p) for f in args.filters)]
    results = []
    tmpdir = Path(tempfile.mkdtemp(prefix="vla_ladder_run_"))
    for p in files:
        script = tmpdir / (p.stem.replace(".", "_") + ".py")
        script.write_text(lesson_to_script(p), encoding="utf-8")
        t0 = time.time()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
        proc = subprocess.run(
            [sys.executable, str(script)], cwd=tmpdir, env=env, capture_output=True, text=True, timeout=args.timeout
        )
        dt = time.time() - t0
        ok = proc.returncode == 0
        results.append((p, ok, dt))
        flag = "✅" if ok else "❌"
        print(f"{flag} {p.relative_to(LESSONS)}  ({dt:.1f}s)", flush=True)
        if args.verbose and ok:
            print(proc.stdout[-4000:])
        if not ok:
            print(proc.stdout[-3000:])
            print(proc.stderr[-5000:])
    print(f"\n通过 {sum(r[1] for r in results)}/{len(results)}；脚本目录：{tmpdir}")
    if not args.keep and all(r[1] for r in results):
        for f in tmpdir.iterdir():
            f.unlink()
        tmpdir.rmdir()
    sys.exit(0 if all(r[1] for r in results) else 1)


if __name__ == "__main__":
    main()
