from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_local_documentation_links_resolve(path):
    text = path.read_text(encoding="utf-8")
    for target in re.findall(r"!?\[[^\]]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
        parts = urlsplit(target)
        if not parts.scheme and not parts.netloc and parts.path:
            destination = path.parent / unquote(parts.path)
            assert destination.exists(), f"{path.name}: missing link target {target}"


@pytest.mark.parametrize("path", DOCUMENTS, ids=lambda path: path.relative_to(ROOT).as_posix())
def test_documentation_uses_neutral_technical_framing(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\b(?:interview\w*|recruit\w*|hiring)\b", text, re.IGNORECASE)
    assert match is None, f"{path.name}: nontechnical framing {match.group()}"


@pytest.mark.parametrize("path", sorted((ROOT / "docs").rglob("*.pdf")), ids=lambda path: path.name)
def test_documentation_pdfs_have_markdown_counterparts(path):
    assert path.with_suffix(".md").is_file(), f"{path.name}: missing Markdown counterpart"
