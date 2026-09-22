import json

import pytest

pytest.importorskip("pypandoc", reason="Install the optional report extra.")
pytest.importorskip("pymupdf", reason="Install the optional report extra.")
pytest.importorskip("typst", reason="Install the optional report extra.")

from scripts.build_pdf_report import (
    SUMMARY_COVER, SUMMARY_SOURCES, assemble, pandoc, walk, wrap_powershell,
)


def test_powershell_commands_wrap_without_changing_their_arguments():
    command = (
        r".\.venv\Scripts\python.exe -m home_valuation pipeline --seed 42 "
        r"--bootstrap 30 --monte-carlo 30 --write-report --output outputs\reference"
    )
    wrapped = wrap_powershell(command)
    assert " `\n" in wrapped
    assert " ".join(wrapped.replace(" `\n", " ").split()) == command
    assert all(len(line) <= 92 for line in wrapped.splitlines())


def test_summary_uses_public_sources_and_resolved_internal_links(tmp_path):
    ast, hashes = assemble(tmp_path)
    headings = {"report-cover"}
    links, mathematics = [], []

    def collect(node):
        kind = node.get("t")
        if kind == "Header":
            headings.add(node["c"][1][0])
        elif kind == "Link":
            links.append(node["c"][-1][0])
        elif kind == "Math":
            mathematics.append(node)
        return node

    walk(ast["blocks"], collect)
    assert all(target[1:] in headings for target in links if target.startswith("#"))
    assert set(path for _, _, path in SUMMARY_SOURCES).issubset(hashes)
    assert all(path == "README.md" or path.startswith("docs/") for path in hashes)
    assert "\\over" not in pandoc(json.dumps(ast), "json", "typst")


def test_summary_is_neutral_two_sided_and_limits_instrument_detail(tmp_path):
    ast, hashes = assemble(tmp_path)
    prose = walk(ast, lambda node: {"t": "Str", "c": ""} if node.get("t") == "Math" else node)
    text = pandoc(json.dumps(prose), "json", "plain").lower()
    assert set(hashes) == {SUMMARY_SOURCES[0][2]}
    for phrase in ["valuation distribution", "seller signal", "buyer signal",
                   "upward escalation", "downward markdown", "risk", "capital"]:
        assert phrase in text
    for phrase in ["interview", "central question", "anderson-rubin", "2sls"]:
        assert phrase not in text
        assert phrase not in SUMMARY_COVER.lower()
    assert text.count("instrumental variables") == 1
    assert "handbook" not in text
    assert "handbook" not in SUMMARY_COVER.lower()
    assert "Valuation ≠ purchase offer ≠ resale list price" in SUMMARY_COVER
    assert len(text.split()) < 4000
    assert "\\over" not in pandoc(json.dumps(ast), "json", "typst")


def test_summary_build_writes_verified_pdf_and_manifest(tmp_path, monkeypatch):
    from scripts import build_pdf_report as builder

    source = builder.local_path(SUMMARY_SOURCES[0][2]).read_bytes()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "design_implementation_summary.md").write_bytes(source)
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["build_pdf_report", "--date", "2026-09-21"])
    builder.main()
    result = docs / "design_implementation_summary.pdf"
    checks = builder.verify_pdf(result)
    assert checks["pages"] <= 12
    manifest = json.loads((tmp_path / "outputs" / "pdf_summary" / "manifest.json").read_text())
    assert manifest["edition"] == "summary"
