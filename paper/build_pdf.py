"""Build paper/paper.pdf from paper/paper.md — pure Python, no pandoc or LaTeX.

Uses python-markdown to render the report to HTML and fpdf2 to lay it out, with
matplotlib's bundled DejaVu fonts (added directly from file, so mathematical glyphs
like Σ, ω, R², →, × render) and the phase figures embedded. Run:

    python paper/build_pdf.py
"""
from __future__ import annotations

import re
from pathlib import Path

import markdown
import matplotlib
from fpdf import FPDF
from fpdf.enums import XPos, YPos

HERE = Path(__file__).resolve().parent
MD = HERE / "paper.md"
PDF = HERE / "paper.pdf"
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
IMG_W = 168  # figure width in mm (content width is ~170mm on A4 with 20mm margins)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        for line in fm.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip().strip('"')
        return meta, body
    return meta, text


def build_html(body: str) -> str:
    html = markdown.markdown(body, extensions=["tables", "fenced_code", "sane_lists"])
    html = html.replace("<strong>", "<b>").replace("</strong>", "</b>")
    html = html.replace("<em>", "<i>").replace("</em>", "</i>")
    # unwrap inline <code> (it would render in a non-Unicode monospace core font);
    # the "code" spans here are maths like Σ_k cos(...), fine in the DejaVu body font
    html = re.sub(r"</?code>", "", html)

    def img(m: re.Match) -> str:
        tag = m.group(0)
        alt = re.search(r'alt="([^"]*)"', tag)
        src = re.search(r'src="([^"]*)"', tag)
        alt_t = alt.group(1) if alt else ""
        abs_src = (HERE / (src.group(1) if src else "")).resolve().as_posix()
        # mm -> px at fpdf2's default 96 dpi for the <img> width attribute
        px = round(IMG_W / 25.4 * 96)
        return (f'<p><img src="{abs_src}" width="{px}"></p>'
                f'<p><font size="8"><i>{alt_t}</i></font></p>')

    return re.sub(r"<p>\s*<img[^>]*>\s*</p>", img, html)


def main() -> None:
    meta, body = parse_front_matter(MD.read_text(encoding="utf-8"))

    pdf = FPDF(format="A4")
    pdf.set_margins(20, 18, 20)
    pdf.set_auto_page_break(True, margin=18)
    for style, fn in [("", "DejaVuSans.ttf"), ("B", "DejaVuSans-Bold.ttf"),
                      ("I", "DejaVuSans-Oblique.ttf"), ("BI", "DejaVuSans-BoldOblique.ttf")]:
        pdf.add_font("DejaVu", style, str(FONTS / fn))
    pdf.add_page()

    # title block
    nl = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}
    pdf.set_font("DejaVu", "B", 18)
    pdf.multi_cell(0, 9, meta.get("title", ""), align="C", **nl)
    pdf.set_font("DejaVu", "", 12)
    pdf.multi_cell(0, 6, meta.get("subtitle", ""), align="C", **nl)
    pdf.set_font("DejaVu", "", 10)
    pdf.multi_cell(0, 5, f"{meta.get('author', '')}  ·  {meta.get('date', '')}", align="C", **nl)
    pdf.ln(4)

    pdf.set_font("DejaVu", "", 10.5)
    pdf.write_html(build_html(body))
    pdf.output(str(PDF))
    print(f"wrote {PDF}  ({PDF.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
