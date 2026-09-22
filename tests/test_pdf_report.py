import json

import pytest

pytest.importorskip("pypandoc", reason="Install the optional report extra.")
pytest.importorskip("pymupdf", reason="Install the optional report extra.")
pytest.importorskip("typst", reason="Install the optional report extra.")

from scripts.build_pdf_report import (
    SOURCES, SUMMARY_COVER, SUMMARY_SOURCES, assemble, metadata_table, pandoc, split_table, text_content, walk,
    wrap_powershell,
)


def test_wide_table_panels_preserve_all_fields_and_values():
    markdown = (
        "| id | alpha | beta | gamma | delta | epsilon | zeta |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        "| row_one | 11 | 22 | 33 | 44 | 55 | 66 |\n"
        "| row_two | 111 | 222 | 333 | 444 | 555 | 666 |\n"
    )
    original = json.loads(pandoc(markdown, "markdown", "json"))["blocks"][0]
    panels = [node for node in split_table(original, "Example") if node["t"] == "Table"]
    assert len(panels) == 2
    headers, rows = [], {"row_one": [], "row_two": []}
    for panel in panels:
        assert len(panel["c"][2]) <= 5
        headers.extend(text_content(c[4]) for c in panel["c"][3][1][0][1][1:])
        for row in panel["c"][4][0][3]:
            key = text_content(row[1][0][4]).replace("\u200b", "")
            rows[key].extend(text_content(c[4]) for c in row[1][1:])
    assert headers == ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    assert rows["row_one"] == ["11", "22", "33", "44", "55", "66"]
    assert rows["row_two"] == ["111", "222", "333", "444", "555", "666"]


def test_readable_metadata_and_command_wrapping():
    table = metadata_table({"seed": 42, "limits": ["synthetic only", "not a guarantee"]})
    rows = table["c"][4][0][3]
    assert [text_content(row[1][0][4]) for row in rows] == ["seed", "limits.1", "limits.2"]
    command = (
        r".\.venv\Scripts\python.exe -m home_valuation pipeline --seed 42 "
        r"--bootstrap 30 --monte-carlo 30 --write-report --output outputs\reference"
    )
    wrapped = wrap_powershell(command)
    assert " `\n" in wrapped
    assert " ".join(wrapped.replace(" `\n", " ").split()) == command
    assert all(len(line) <= 92 for line in wrapped.splitlines())


def test_book_uses_public_sources_and_resolved_internal_links(tmp_path):
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
    assert len(mathematics) > 40
    assert set(path for _, _, path in SOURCES).issubset(hashes)
    assert all(path == "README.md" or path.startswith("docs/") for path in hashes)
    assert len(list((tmp_path / "assets").glob("*.png"))) == 6
    assert [source[0] for source in SOURCES][-2:] == ["linked", "linked-results"]
    assert "\\over" not in pandoc(json.dumps(ast), "json", "typst")


def test_summary_is_neutral_two_sided_and_limits_instrument_detail(tmp_path):
    ast, hashes = assemble(tmp_path, summary=True)
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
    assert "Valuation ≠ purchase offer ≠ resale list price" in SUMMARY_COVER
    assert len(text.split()) < 4000
    assert "\\over" not in pandoc(json.dumps(ast), "json", "typst")


def test_summary_build_preserves_the_full_handbook(tmp_path, monkeypatch):
    from scripts import build_pdf_report as builder

    source = builder.local_path(SUMMARY_SOURCES[0][2]).read_bytes()
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "design_implementation_summary.md").write_bytes(source)
    full = docs / "design_implementation_report.pdf"
    full.write_bytes(b"preserved existing handbook")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr("sys.argv", ["build_pdf_report", "--summary", "--date", "2026-09-21"])
    builder.main()
    assert full.read_bytes() == b"preserved existing handbook"
    result = docs / "design_implementation_summary.pdf"
    checks = builder.verify_pdf(result, summary=True)
    assert checks["pages"] <= 12
    manifest = json.loads((tmp_path / "outputs" / "pdf_summary" / "manifest.json").read_text())
    assert manifest["edition"] == "summary"
