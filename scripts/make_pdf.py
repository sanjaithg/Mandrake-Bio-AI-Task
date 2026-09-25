"""Lightweight Markdown -> PDF renderer (reportlab), tailored to REPORT.md's
subset of Markdown: headers, paragraphs, bullet/numbered lists, fenced code
blocks, tables, bold/inline-code spans, horizontal rules. Good enough for
this doc; not a general Markdown engine.
"""
import os
import re
import sys
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, ListFlowable,
                                 ListItem, Preformatted, Table, TableStyle, HRFlowable,
                                 Image, PageBreak)
from PIL import Image as PILImage

def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier">\1</font>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<u>\1</u>', text)
    return text

def build(md_path, pdf_path, title, compact=False, author=None, images=None):
    styles = getSampleStyleSheet()
    if compact:
        h1size, h2size, bodysize, leading, margin = 14, 11, 8.3, 10.6, 0.55*inch
    else:
        h1size, h2size, bodysize, leading, margin = 17, 13.5, 9.7, 13.5, 0.75*inch
    styles.add(ParagraphStyle("H1c", parent=styles["Heading1"], spaceBefore=8 if compact else 14, spaceAfter=4 if compact else 8, fontSize=h1size))
    styles.add(ParagraphStyle("H2c", parent=styles["Heading2"], spaceBefore=7 if compact else 12, spaceAfter=3 if compact else 6, fontSize=h2size))
    styles.add(ParagraphStyle("Body", parent=styles["Normal"], fontSize=bodysize, leading=leading, spaceAfter=3 if compact else 6))
    styles.add(ParagraphStyle("BulletItem", parent=styles["Body"], leftIndent=12 if compact else 14, spaceAfter=1 if compact else 3))
    styles.add(ParagraphStyle("CodeBlock", parent=styles["Code"], fontSize=6.6 if compact else 7.6, leading=8 if compact else 9.5, backColor=colors.whitesmoke))
    styles.add(ParagraphStyle("Cover", parent=styles["Title"], fontSize=15 if compact else 22, spaceAfter=2))
    styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=8.5 if compact else 11, textColor=colors.grey, spaceAfter=6 if compact else 20))
    styles.add(ParagraphStyle("Caption", parent=styles["Normal"], fontSize=7.5 if compact else 9, textColor=colors.grey, spaceAfter=10, alignment=1))

    doc = SimpleDocTemplate(pdf_path, pagesize=LETTER,
                             topMargin=margin, bottomMargin=margin,
                             leftMargin=margin, rightMargin=margin,
                             title=title, author=author or "")
    content_width = LETTER[0] - 2 * margin

    subtitle = "Mandrake AI Bio Research — Cas9 Assignment Pack"
    if author:
        subtitle = f"{author} · {subtitle}"
    story = [Paragraph(title, styles["Cover"]),
             Paragraph(subtitle, styles["CoverSub"]),
             HRFlowable(width="100%", color=colors.HexColor("#888888"), thickness=0.5 if compact else 0.75), Spacer(1, 4 if compact else 10)]

    lines = open(md_path, encoding="utf-8").read().splitlines()
    i = 0
    list_buffer = []
    list_ordered = False

    def flush_list():
        nonlocal list_buffer, list_ordered
        if list_buffer:
            items = [ListItem(Paragraph(inline(t), styles["BulletItem"])) for t in list_buffer]
            story.append(ListFlowable(items, bulletType="1" if list_ordered else "bullet",
                                       leftIndent=16, spaceBefore=2, spaceAfter=8))
            list_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_list()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            story.append(Preformatted("\n".join(code_lines), styles["CodeBlock"]))
            story.append(Spacer(1, 8))
            i += 1
            continue

        if stripped.startswith("# "):
            flush_list(); story.append(Paragraph(inline(stripped[2:]), styles["H1c"])); i += 1; continue
        if stripped.startswith("## "):
            flush_list(); story.append(Paragraph(inline(stripped[3:]), styles["H2c"])); i += 1; continue
        if stripped.startswith("### "):
            flush_list(); story.append(Paragraph(inline(stripped[4:]), styles["H2c"])); i += 1; continue

        if stripped.startswith("|"):
            flush_list()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            rows = []
            for tl in table_lines:
                if re.match(r"^\|[\s:\-|]+\|$", tl):
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows.append([Paragraph(inline(c), styles["Body"]) for c in cells])
            if rows:
                t = Table(rows, hAlign="LEFT")
                t.setStyle(TableStyle([
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]))
                story.append(t)
                story.append(Spacer(1, 10))
            continue

        if re.match(r"^[-*]\s+", stripped):
            list_ordered = False
            list_buffer.append(re.sub(r"^[-*]\s+", "", stripped))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            list_ordered = True
            list_buffer.append(re.sub(r"^\d+\.\s+", "", stripped))
            i += 1
            continue

        if stripped == "":
            flush_list()
            i += 1
            continue

        flush_list()
        story.append(Paragraph(inline(stripped), styles["Body"]))
        i += 1

    flush_list()

    if images:
        story.append(PageBreak())
        story.append(Paragraph("Figures", styles["H1c"]))
        for img_path in images:
            with PILImage.open(img_path) as im:
                w, h = im.size
            draw_w = content_width
            draw_h = draw_w * h / w
            max_h = doc.height * 0.85
            if draw_h > max_h:
                draw_h = max_h
                draw_w = draw_h * w / h
            story.append(Image(img_path, width=draw_w, height=draw_h))
            story.append(Paragraph(inline(os.path.basename(img_path)), styles["Caption"]))

    doc.build(story)
    print(f"Wrote {pdf_path}")

if __name__ == "__main__":
    md_path = sys.argv[1]
    pdf_path = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else "Document"
    compact = "--compact" in sys.argv
    author = None
    if "--author" in sys.argv:
        author = sys.argv[sys.argv.index("--author") + 1]
    images = None
    if "--images" in sys.argv:
        images = sys.argv[sys.argv.index("--images") + 1].split(",")
    build(md_path, pdf_path, title, compact=compact, author=author, images=images)
