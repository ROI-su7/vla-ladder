"""把 lessons/ 下的 Markdown 课程源文件构建成三种产物。

一份源文件 -> 三种用法：
  1. GitHub 上直接阅读：lessons/*.md 本身（公式、可折叠答案都能正常显示）
  2. 动手写代码：notebooks/*.ipynb（Colab / Jupyter 打开，答案单元默认折叠）
  3. 在线文档网站：build/docs/*.md -> mkdocs 生成 GitHub Pages

课程源文件的约定（写新课时请遵守）：
  * 第一行是 `# 标题`
  * ```python 代码块 -> notebook 中的代码单元
  * 紧跟在练习代码块之后的
        <details>
        <summary>💡 点击查看参考答案</summary>

        ```python
        ...
        ```

        </details>
    -> 参考答案（notebook 中折叠显示；网站中可点击展开）
  * 其它语言的代码块（```bash / ```text 等）保持为普通 Markdown

用法：
  python tools/build.py            # 生成 notebooks/ 与 build/docs/，并刷新 mkdocs.yml 的导航
  python tools/build.py --check    # 只做解析检查
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "lessons"
NOTEBOOKS = ROOT / "notebooks"
DOCS = ROOT / "build" / "docs"

GITHUB_USER = "ROI-su7"
REPO_NAME = "vla-ladder"
BRANCH = "main"
COLAB_PREFIX = f"https://colab.research.google.com/github/{GITHUB_USER}/{REPO_NAME}/blob/{BRANCH}/notebooks"
BADGE_IMG = "https://colab.research.google.com/assets/colab-badge.svg"
SOLUTION_TITLE = "💡 参考答案（点击展开查看；运行本单元即可使用参考实现）"

# 每个 notebook 开头自动插入的环境准备单元
SETUP_CODE = f'''# 🔧 环境准备（每个 notebook 都先运行这一格）
# - 在 Colab 中：自动下载本项目代码
# - 在本地：自动把项目根目录加入 sys.path（也可以在根目录执行 `pip install -e .`）
import os, sys
if "google.colab" in sys.modules:
    if not os.path.exists("/content/{REPO_NAME}"):
        os.system("git clone -q --depth 1 https://github.com/{GITHUB_USER}/{REPO_NAME}.git /content/{REPO_NAME}")
    sys.path.insert(0, "/content/{REPO_NAME}")
else:
    _p = os.path.abspath(".")
    while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "vla_ladder")):
        _p = os.path.dirname(_p)
    sys.path.insert(0, _p)
print("环境就绪 ✅")'''


@dataclass
class Cell:
    kind: str  # "markdown" | "code" | "solution"
    source: str
    meta: dict = field(default_factory=dict)


def parse_lesson(text: str) -> list[Cell]:
    """把课程 Markdown 切成 markdown / code / solution 三类单元。"""
    lines = text.splitlines()
    cells: list[Cell] = []
    md_buf: list[str] = []
    i = 0

    def flush_md():
        content = "\n".join(md_buf).strip("\n")
        if content.strip():
            cells.append(Cell("markdown", content))
        md_buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # ---- 顶层 python 代码块 -> code 单元 ----
        if stripped == "```python" and not line.startswith(" "):
            flush_md()
            j = i + 1
            code: list[str] = []
            while j < len(lines) and lines[j].strip() != "```":
                code.append(lines[j])
                j += 1
            if j >= len(lines):
                raise ValueError(f"第 {i + 1} 行的 ```python 代码块没有闭合")
            cells.append(Cell("code", "\n".join(code)))
            i = j + 1
            continue
        # ---- <details> 参考答案 -> solution 单元 ----
        if stripped.startswith("<details"):
            j = i + 1
            block: list[str] = []
            while j < len(lines) and not lines[j].strip().startswith("</details>"):
                block.append(lines[j])
                j += 1
            if j >= len(lines):
                raise ValueError(f"第 {i + 1} 行的 <details> 没有闭合")
            summary = ""
            code: list[str] = []
            notes: list[str] = []
            in_code = False
            for b in block:
                bs = b.strip()
                if bs.startswith("<summary>"):
                    summary = re.sub(r"</?summary>", "", bs).strip()
                elif bs == "```python" and not in_code:
                    in_code = True
                elif bs == "```" and in_code:
                    in_code = False
                elif in_code:
                    code.append(b)
                else:
                    notes.append(b)
            if code:
                flush_md()
                cells.append(Cell("solution", "\n".join(code), {"summary": summary, "notes": "\n".join(notes).strip()}))
            else:  # 不含代码的 details（例如“思考题答案”）原样保留为 markdown
                md_buf.extend(lines[i : j + 1])
            i = j + 1
            continue
        md_buf.append(line)
        i += 1
    flush_md()
    return cells


def lesson_files() -> list[Path]:
    return sorted(p for p in LESSONS.rglob("*.md"))


def rel(p: Path) -> Path:
    return p.relative_to(LESSONS)


def colab_url(lesson: Path) -> str:
    return f"{COLAB_PREFIX}/{rel(lesson).with_suffix('.ipynb').as_posix()}"


def is_index(lesson: Path) -> bool:
    return lesson.name in ("README.md", "index.md")


def badge_line(lesson: Path) -> str:
    return f"[![在 Colab 中打开]({BADGE_IMG})]({colab_url(lesson)})"


def ensure_badge(lesson: Path) -> None:
    """保证每篇课程标题下方有正确的 Colab 徽章（GitHub 阅读时也能一键打开练习）。"""
    if is_index(lesson):
        return
    text = lesson.read_text(encoding="utf-8")
    lines = text.splitlines()
    want = badge_line(lesson)
    if len(lines) > 2 and "colab-badge.svg" in lines[2]:
        if lines[2] != want:
            lines[2] = want
            lesson.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    if not lines or not lines[0].startswith("# "):
        raise ValueError(f"{lesson} 第一行必须是 '# 标题'")
    lines = [lines[0], "", want, ""] + lines[1:]
    lesson.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- notebook


def to_notebook(lesson: Path, cells: list[Cell]) -> dict:
    nb_cells = []

    def src(s: str) -> list[str]:
        parts = s.split("\n")
        return [p + "\n" for p in parts[:-1]] + [parts[-1]]

    inserted_setup = False
    for c in cells:
        if c.kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src(c.source)})
            if not inserted_setup:  # 标题之后插入环境准备单元
                nb_cells.append(
                    {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(SETUP_CODE)}
                )
                inserted_setup = True
        elif c.kind == "code":
            nb_cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src(c.source)})
        else:  # solution
            body = f'#@title {SOLUTION_TITLE} {{ display-mode: "form" }}\n' + c.source
            nb_cells.append(
                {
                    "cell_type": "code",
                    "metadata": {"cellView": "form", "jupyter": {"source_hidden": True}, "tags": ["solution"]},
                    "execution_count": None,
                    "outputs": [],
                    "source": src(body),
                }
            )
    for k, cell in enumerate(nb_cells):  # nbformat 4.5 要求每个单元有 id
        cell["id"] = f"cell-{k:03d}"
    return {
        "cells": nb_cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ----------------------------------------------------------------------------- docs site


def to_docs_md(lesson: Path, cells: list[Cell]) -> str:
    out: list[str] = []
    for c in cells:
        if c.kind == "markdown":
            out.append(c.source)
        elif c.kind == "code":
            out.append("```python\n" + c.source + "\n```")
        else:
            title = c.meta.get("summary") or "💡 点击查看参考答案"
            body = "```python\n" + c.source + "\n```"
            if c.meta.get("notes"):
                body = c.meta["notes"] + "\n\n" + body
            indented = "\n".join(("    " + ln) if ln.strip() else "" for ln in body.splitlines())
            out.append(f'??? success "{title}"\n\n{indented}')
    text = "\n\n".join(out) + "\n"
    # 站内链接：章节的 README.md 在网站中是 index.md
    text = re.sub(r"\]\(((?:\.\./)?[\w.]*/?)README\.md\)", r"](\1index.md)", text)
    # 指向仓库代码的相对链接改成 GitHub 绝对链接
    text = re.sub(
        r"\]\((\.\./)+(vla_ladder/[^)]+)\)",
        lambda m: f"](https://github.com/{GITHUB_USER}/{REPO_NAME}/blob/{BRANCH}/{m.group(2)})",
        text,
    )
    return text


def title_of(p: Path) -> str:
    first = p.read_text(encoding="utf-8").splitlines()[0]
    return first.lstrip("# ").strip()


def build_nav() -> list:
    """按目录顺序生成 mkdocs 导航。"""
    nav: list = [{"首页": "index.md"}]
    for chapter in sorted(d for d in LESSONS.iterdir() if d.is_dir()):
        items: list = []
        idx = chapter / "README.md"
        if idx.exists():
            items.append({"本章导读": f"{chapter.name}/index.md"})
        for p in sorted(chapter.glob("*.md")):
            if is_index(p):
                continue
            items.append({title_of(p): f"{chapter.name}/{p.name}"})
        nav.append({title_of(idx) if idx.exists() else chapter.name: items})
    return nav


def write_mkdocs_nav(nav: list) -> None:
    import yaml  # 只在构建网站时需要

    cfg_path = ROOT / "mkdocs.yml"
    text = cfg_path.read_text(encoding="utf-8")
    marker = "# ---- nav（由 tools/build.py 自动生成，请勿手改）----"
    head = text.split(marker)[0].rstrip() + "\n\n"
    nav_yaml = yaml.safe_dump({"nav": nav}, allow_unicode=True, sort_keys=False)
    cfg_path.write_text(head + marker + "\n" + nav_yaml, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只解析检查，不写文件")
    ap.add_argument("--no-docs", action="store_true", help="不生成网站文件")
    args = ap.parse_args()

    files = lesson_files()
    n_code = n_sol = 0
    for f in files:
        cells = parse_lesson(f.read_text(encoding="utf-8"))
        n_code += sum(c.kind == "code" for c in cells)
        n_sol += sum(c.kind == "solution" for c in cells)
    print(f"共 {len(files)} 个课程文件，{n_code} 个代码单元，{n_sol} 个参考答案")
    if args.check:
        return

    if NOTEBOOKS.exists():
        shutil.rmtree(NOTEBOOKS)
    if DOCS.exists():
        shutil.rmtree(DOCS)

    for f in files:
        ensure_badge(f)
        cells = parse_lesson(f.read_text(encoding="utf-8"))
        r = rel(f)
        if not is_index(f):
            nb_path = NOTEBOOKS / r.with_suffix(".ipynb")
            nb_path.parent.mkdir(parents=True, exist_ok=True)
            nb_path.write_text(json.dumps(to_notebook(f, cells), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        if not args.no_docs:
            out = DOCS / (r.parent / "index.md" if is_index(f) else r)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(to_docs_md(f, cells), encoding="utf-8")

    if not args.no_docs:
        # 网站首页 = 仓库 README；链接从 lessons/xxx.md 改写为 xxx.md
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme = re.sub(r"\]\(lessons/([^)]*)README\.md\)", r"](\1index.md)", readme)
        readme = re.sub(r"\]\(lessons/", "](", readme)
        readme = re.sub(r"\]\((vla_ladder/[^)]+|tools/[^)]+)\)", rf"](https://github.com/{GITHUB_USER}/{REPO_NAME}/blob/{BRANCH}/\1)", readme)
        (DOCS / "index.md").write_text(readme, encoding="utf-8")
        assets = ROOT / "assets"
        if assets.exists():
            shutil.copytree(assets, DOCS / "assets")
        extra = ROOT / "tools" / "site_extra"
        if extra.exists():
            shutil.copytree(extra, DOCS, dirs_exist_ok=True)
        write_mkdocs_nav(build_nav())
    print("构建完成 ✅")


if __name__ == "__main__":
    main()
