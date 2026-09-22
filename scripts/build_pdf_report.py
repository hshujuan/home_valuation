"""Build the full handbook or separate technical summary from public Markdown."""

import argparse
from copy import deepcopy
from datetime import date
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import textwrap
from urllib.parse import unquote, urlsplit

import pypandoc
import pymupdf
import typst


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ("guide", "I. Design and implementation guide", "docs/design_implementation_guide.md"),
    ("analysis", "II. Theory and economic reasoning", "docs/analysis_report.md"),
    ("contracts", "III. Data and accounting contracts", "docs/data_dictionary.md"),
    ("results", "IV. Reference results", "docs/reference_results.md"),
    ("runbook", "V. Practical runbook", "README.md"),
    ("sources", "VI. Sources and evidence", "docs/sources.md"),
    ("linked", "VII. Linked two-sided extension", "docs/two_sided_extension.md"),
    ("linked-results", "VIII. Linked two-sided results", "docs/two_sided_results.md"),
)
SUMMARY_SOURCES = (
    ("summary", "Technical synthesis", "docs/design_implementation_summary.md"),
)
SUMMARY_COVER = r"""
#v(0.6in)
#text(size: 10pt, weight: "bold", fill: accent)[TECHNICAL SUMMARY]
#v(0.3in)
#text(size: 32pt, weight: "bold", fill: ink)[Home Valuation\
and Two-sided\
Causal Decisions]
<report-cover>
#v(0.2in)
#text(size: 17pt, fill: muted)[Valuation, behavioral response,\
and portfolio economics]
#v(0.3in)
#line(length: 100%, stroke: 2pt + accent)
#v(0.2in)
#block(fill: rgb("#e7eff5"), inset: 14pt, radius: 4pt)[
  *Valuation ≠ purchase offer ≠ resale list price*
]
#v(0.2in)
#text(size: 12pt)[
  A two-sided decision system built on a valuation distribution:\
  acquisition offers, seller signals, and upward escalation;\
  resale listings, buyer signals, and downward markdown;\
  lifecycle economics, risk, and capital allocation.
]
#v(0.25in)
*Scope:* public context, explicit modeling assumptions, and synthetic evidence.\
*Status:* an implemented prototype and proposed extensions, not a description
of a company's internal production system.\
*Snapshot:* SNAPSHOT_DATE.
#v(0.2in)
#text(size: 9pt, fill: muted)[
  A concise, standalone synthesis. The complete implementation handbook is
  retained separately and is not replaced by this document.
]
#pagebreak()
#text(size: 22pt, weight: "bold", fill: ink)[Contents]
#v(0.2in)
#outline(title: none, depth: 2, indent: 1em)
"""
STYLE = r"""
#let ink = rgb("#17324a")
#let accent = rgb("#167b83")
#let muted = rgb("#566879")
#set document(
  title: "Home Valuation and Causal Pricing: Design and Implementation",
  author: "Home Valuation Lab",
)
#set page(
  paper: "us-letter",
  margin: (left: 0.75in, right: 0.75in, top: 0.72in, bottom: 0.65in),
  header: context if counter(page).get().first() > 1 {
    text(size: 8pt, fill: muted)[HOME VALUATION / CAUSAL PRICING / IMPLEMENTATION]
  },
  footer: context align(right)[
    #text(size: 8pt, fill: muted)[
      Synthetic research handbook
      #h(1em)
      #counter(page).display("1") / #counter(page).final().first()
    ]
  ],
)
#set text(font: "Libertinus Serif", size: 10.5pt, lang: "en")
#set par(justify: false, leading: 0.55em, spacing: 0.65em)
#set heading(numbering: none)
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  text(size: 21pt, weight: "bold", fill: ink, it)
}
#show heading.where(level: 2): set text(size: 14pt, fill: ink)
#show heading.where(level: 3): set text(size: 11.5pt, fill: accent)
#show link: set text(fill: accent)
#set table(
  inset: 5pt,
  stroke: (bottom: 0.3pt + rgb("#d8e1e8")),
  fill: (x, y) => if y == 0 { rgb("#e7eff5") }
    else if calc.odd(y) { rgb("#f6f8fa") } else { white },
)
#show table: set text(size: 9pt)
#show table.cell.where(y: 0): set text(weight: "bold", fill: ink)
#show figure.where(kind: table): set figure(supplement: none)
#show figure.where(kind: table): set block(breakable: true)
#show figure.where(kind: image): set figure(supplement: none)
#show figure.caption: set text(size: 9pt, fill: muted)
#show raw: set text(font: "DejaVu Sans Mono", size: 8.1pt)
#show raw.where(block: true): it => block(
  width: 100%, inset: 8pt, fill: rgb("#f2f5f8"), radius: 3pt, breakable: true, it,
)
#show quote.where(block: true): set text(fill: muted)
#let horizontalrule = line(length: 100%, stroke: 0.6pt + rgb("#d8e1e8"))
"""


def local_path(relative):
    return ROOT.joinpath(*relative.split("/"))


def pandoc(value, source, target):
    result = subprocess.run(
        [pypandoc.get_pandoc_path(), "--from", source, "--to", target,
         "--syntax-highlighting=none", "--wrap=none"],
        input=value, text=True, encoding="utf-8", capture_output=True, check=True,
    )
    if result.stderr.strip():
        raise RuntimeError(f"Pandoc reported a conversion warning:\n{result.stderr}")
    return result.stdout


def walk(value, transform):
    if isinstance(value, list):
        return [walk(item, transform) for item in value]
    if isinstance(value, dict):
        value = {key: walk(item, transform) for key, item in value.items()}
        return transform(value)
    return value


def text_content(value):
    if isinstance(value, list):
        return "".join(text_content(item) for item in value)
    if not isinstance(value, dict):
        return ""
    kind = value.get("t")
    if kind == "Str":
        return value["c"]
    if kind in {"Space", "SoftBreak", "LineBreak"}:
        return " "
    if kind == "Code":
        return value["c"][1]
    return text_content(value.get("c", []))


def plain(text):
    return {"t": "Plain", "c": [{"t": "Str", "c": text}]}


def cell(text):
    return [["", [], []], {"t": "AlignDefault"}, 1, 1, [plain(text)]]


def metadata_table(data):
    rows = []

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                flatten(child, f"{prefix}.{key}" if prefix else key)
        elif isinstance(value, list):
            for index, child in enumerate(value, 1):
                flatten(child, f"{prefix}.{index}")
        else:
            rows.append([["", [], []], [cell(prefix), cell(str(value))]])

    flatten(data)
    return {"t": "Table", "c": [
        ["", [], []], [None, []],
        [[{"t": "AlignDefault"}, {"t": "ColWidth", "c": weight}] for weight in (0.38, 0.62)],
        [["", [], []], [[["", [], []], [cell("Metadata field"), cell("Value")]]]],
        [[["", [], []], 0, [], rows]],
        [["", [], []], []],
    ]}


def readable_table(table):
    for column in table["c"][2]:
        if column[0]["t"] == "AlignDefault":
            column[0] = {"t": "AlignLeft"}

    def soften(node):
        if node.get("t") == "Str":
            for separator in ("_", "/", "\\", "."):
                node["c"] = node["c"].replace(separator, separator + "\u200b")
        elif node.get("t") == "Code":
            text = node["c"][1]
            for separator in ("_", "/", "\\", "."):
                text = text.replace(separator, separator + "\u200b")
            return {"t": "Str", "c": text}
        return node
    return walk(table, soften)


def split_table(table, section):
    columns = len(table["c"][2])
    if columns <= 5:
        return [readable_table(table)]
    keys = {
        "Valuation Metrics": [0, 1], "Valuation Slices": [0, 1],
        "Causal Effects": [6, 0], "Stress Curves": [9, 0],
        "Iv Exclusion Sensitivity": [6, 0],
        "Stage Effects": [0, 1],
    }.get(section, [0])
    metrics = [index for index in range(columns) if index not in keys]
    size = 5 - len(keys)
    panels = []
    for start in range(0, len(metrics), size):
        selected = keys + metrics[start:start + size]
        panel = deepcopy(table)
        content = panel["c"]
        content[2] = [[{"t": "AlignDefault"}, {"t": "ColWidthDefault"}] for _ in selected]
        for part in [content[3], content[5]]:
            for row in part[1]:
                row[1] = [row[1][index] for index in selected]
        for body in content[4]:
            for row in body[2] + body[3]:
                row[1] = [row[1][index] for index in selected]
        panels.append(plain(f"Column panel {start // size + 1}; row identifiers are repeated."))
        panels.append(readable_table(panel))
    return panels


def wrap_powershell(text):
    lines = []
    for line in text.splitlines():
        if len(line) <= 92:
            lines.append(line)
        elif line.startswith("#"):
            lines.extend(textwrap.wrap(line[1:].strip(), width=88,
                                       initial_indent="# ", subsequent_indent="# "))
        else:
            pieces = textwrap.wrap(line, width=88, subsequent_indent="    ",
                                   break_long_words=False, break_on_hyphens=False)
            lines.append(" `\n".join(pieces))
    return "\n".join(lines)


def assemble(build, summary=False):
    sources = SUMMARY_SOURCES if summary else SOURCES
    chapter_ids = {local_path(path).resolve(): key for key, _, path in sources}
    filename = "design_implementation_summary.pdf" if summary else "design_implementation_report.pdf"
    output_pdf = ROOT / "docs" / filename
    companion_pdfs = {ROOT / "docs" / name for name in
                      ["design_implementation_report.pdf", "design_implementation_summary.pdf"]}
    all_blocks, hashes, assets = [], {}, {}
    api_version = None
    for key, title, relative in sources:
        path = local_path(relative)
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        parsed = json.loads(pandoc(path.read_text(encoding="utf-8"),
                                   "markdown+tex_math_single_backslash", "json"))
        api_version = parsed["pandoc-api-version"]

        def rewrite(node):
            kind = node.get("t")
            if kind == "Header":
                node["c"][1][0] = key + "-" + node["c"][1][0]
            elif kind in {"Link", "Image"}:
                target = node["c"][-1][0]
                parts = urlsplit(target)
                if not parts.scheme and not parts.netloc:
                    destination = (path.parent / unquote(parts.path)).resolve()
                    if kind == "Image":
                        if not destination.is_relative_to(ROOT / "docs" / "figures"):
                            raise ValueError(f"Image outside the public figure allowlist: {target}")
                        asset = build / "assets" / destination.name
                        asset.parent.mkdir(exist_ok=True)
                        shutil.copyfile(destination, asset)
                        assets[destination.relative_to(ROOT).as_posix()] = hashlib.sha256(
                            destination.read_bytes()).hexdigest()
                        node["c"][-1][0] = "assets/" + destination.name
                    elif not parts.path:
                        node["c"][-1][0] = "#" + key + "-" + parts.fragment
                    elif destination == output_pdf:
                        node["c"][-1][0] = "#report-cover"
                    elif destination in companion_pdfs:
                        node["c"][-1][0] = destination.name
                    elif destination in chapter_ids:
                        anchor = chapter_ids[destination]
                        node["c"][-1][0] = "#" + anchor + (
                            "-" + parts.fragment if parts.fragment else "")
                    else:
                        raise ValueError(f"Local link is not included in the PDF: {relative}: {target}")
            elif kind == "CodeBlock":
                classes = node["c"][0][1]
                if "powershell" in classes:
                    node["c"][1] = wrap_powershell(node["c"][1])
            return node

        blocks = walk(parsed["blocks"], rewrite)
        if blocks and blocks[0].get("t") == "Header" and blocks[0]["c"][0] == 1:
            blocks.pop(0)
        all_blocks.append({"t": "Header", "c": [1, [key, [], []], [{"t": "Str", "c": title}]]})
        section = ""
        for block in blocks:
            if block.get("t") == "Header":
                section = text_content(block["c"][2])
            if block.get("t") == "Table":
                all_blocks.extend(split_table(block, section) if key in {"results", "linked-results"}
                                  else [readable_table(block)])
            elif (block.get("t") == "CodeBlock"
                  and "json" in block["c"][0][1] and key == "results"):
                all_blocks.append(readable_table(metadata_table(json.loads(block["c"][1]))))
            else:
                all_blocks.append(block)
    return {"pandoc-api-version": api_version, "meta": {}, "blocks": all_blocks}, hashes | assets


def verify_pdf(path, summary=False):
    with pymupdf.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
        normalized = " ".join(text.replace("\u200b", "").split())
        required = ["Design and implementation guide", "Theory and economic reasoning",
                    "Comparative implementation review", "Data and accounting contracts",
                    "Reference results", "Practical runbook", "Scope and acceptance criteria",
                    "Sources and evidence", "Linked two-sided extension", "Linked two-sided results",
                    "17.86%", "25.93%", "86.21%", "exclusion", "sequential"]
        if summary:
            required = ["Decision architecture", "Valuation and underwriting as a distribution",
                        "Acquisition: initial offer", "Resale: initial list",
                        "Causal design across both sides", "Connect the stages through downstream value",
                        "Portfolio economics, risk, and capital", "Illustrative evidence",
                        "Implementation and operating priorities", "Selected references",
                        "Seller signal", "Buyer signal", "86.21%"]
            if len(document) > 12:
                raise ValueError("The technical summary must remain at most twelve pages.")
            for phrase in ["interview", "central question"]:
                if phrase in normalized.lower():
                    raise ValueError(f"Unwanted summary framing: {phrase}")
        for phrase in required:
            if phrase not in normalized:
                raise ValueError(f"Missing PDF content: {phrase}")
        if "\ufffd" in text:
            raise ValueError("The PDF contains replacement glyphs.")
        if not document.get_toc():
            raise ValueError("The PDF has no navigation bookmarks.")
        images = set()
        for page in document:
            images.update(image[0] for image in page.get_images())
            prose_lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    if len("".join(span["text"] for span in line["spans"])) > 25:
                        prose_lines.append(pymupdf.Rect(line["bbox"]))
                    for span in line.get("spans", []):
                        x0, y0, x1, y1 = span["bbox"]
                        if x0 < 30 or y0 < 18 or x1 > page.rect.width - 30 or y1 > page.rect.height - 18:
                            raise ValueError(f"Text outside safe page bounds on page {page.number + 1}")
            for index, left in enumerate(prose_lines):
                for right in prose_lines[index + 1:]:
                    overlap = left & right
                    if (not overlap.is_empty and overlap.width > 25
                            and overlap.height > 0.75 * min(left.height, right.height)):
                        raise ValueError(f"Overlapping text on page {page.number + 1}")
        if not summary and len(images) < 6:
            raise ValueError("Expected the five original figures and the linked-extension figure.")
        return {"pages": len(document), "bookmarks": len(document.get_toc()),
                "embedded_images": len(images)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="ISO snapshot date; defaults to today.")
    parser.add_argument("--summary", action="store_true",
                        help="Build the separate concise summary; leave the full handbook untouched.")
    args = parser.parse_args()
    build = ROOT / "outputs" / ("pdf_summary" if args.summary else "pdf_report")
    build.mkdir(parents=True, exist_ok=True)
    ast, hashes = assemble(build, summary=args.summary)
    (build / "book.json").write_text(json.dumps(ast, ensure_ascii=False), encoding="utf-8")
    body = pandoc(json.dumps(ast), "json", "typst")
    cover = SUMMARY_COVER if args.summary else r"""
#v(1.0in)
#text(size: 10pt, weight: "bold", fill: accent)[RESEARCH + ENGINEERING HANDBOOK]
#v(0.3in)
#text(size: 34pt, weight: "bold", fill: ink)[Home Valuation\
and Causal Pricing]
<report-cover>
#v(0.18in)
#text(size: 19pt, fill: muted)[Design, motivation, theory,\
implementation, and reproducible examples]
#v(0.35in)
#line(length: 100%, stroke: 2pt + accent)
#v(0.2in)
#text(size: 12pt)[
  An end-to-end guide to predicting value, identifying price response,
  and optimizing lifecycle contribution.
]
#v(0.25in)
#block(fill: rgb("#e7eff5"), inset: 14pt, radius: 4pt)[
  *Central question* \
  If we reduce a resale list price by 3%, how does completion change,
  and is that change worth its economic cost?
]
#v(0.3in)
*Scope:* independent synthetic research, not an Opendoor production model. \
*Evidence:* maintained explanations, generated results, and public citations. \
*Snapshot:* SNAPSHOT_DATE; seed-42 reference study.
#v(0.2in)
#text(size: 9pt, fill: muted)[
  The reference study uses 5,000 market transactions, 4,000 seller opportunities,
  6,000 resale decisions per main assignment design, and 8,000 rows per main IV fixture.
  Short examples and app runs use different sample sizes.
  Parts IX-X add a separate linked acquisition/resale experiment with
  12,000 development, 10,000 continuation-learning, and 8,000 evaluation prospects.
]
#pagebreak()
#text(size: 22pt, weight: "bold", fill: ink)[Contents]
#v(0.2in)
#outline(title: none, depth: 2, indent: 1em)
"""
    style = STYLE
    if args.summary:
        style = style.replace("Synthetic research handbook", "Technical summary / Synthetic evidence")
        style += '\n#set document(title: "Home Valuation and Two-sided Causal Decisions: Technical Summary")\n'
    content = style + cover.replace("SNAPSHOT_DATE", args.date.isoformat()) + body
    source = build / "book.typ"
    source.write_text(content, encoding="utf-8")
    filename = ("design_implementation_summary.pdf" if args.summary
                else "design_implementation_report.pdf")
    candidate = build / filename
    typst.compile(str(source), output=str(candidate), root=str(build))
    checks = verify_pdf(candidate, summary=args.summary)
    output = ROOT / "docs" / candidate.name
    shutil.copyfile(candidate, output)
    manifest = {
        "snapshot_date": args.date.isoformat(), "edition": "summary" if args.summary else "full",
        **checks, "inputs_sha256": hashes,
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ("pypandoc-binary", "typst", "PyMuPDF")},
        "pandoc_version": str(pypandoc.get_pandoc_version()),
    }
    (build / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created {output.name}: {checks['pages']} pages, "
          f"{checks['bookmarks']} bookmarks, {checks['embedded_images']} figures.")
    print("Public sources only; no statistical pipeline, private inputs, or network fetches.")


if __name__ == "__main__":
    main()
