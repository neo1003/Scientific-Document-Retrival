from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"

BG_TOP = (242, 237, 227)
BG_BOTTOM = (226, 212, 192)
PANEL = (255, 249, 241, 242)
PANEL_SOFT = (246, 239, 229, 255)
WHITE = (255, 255, 255, 245)
INK = (36, 23, 14)
MUTED = (110, 90, 74)
ACCENT = (31, 109, 98)
ACCENT_DARK = (20, 62, 57)
WARM = (211, 110, 67)
GREEN = (23, 163, 119)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(["georgiab.ttf", "arialbd.ttf", "seguisb.ttf"])
    else:
        candidates.extend(["georgia.ttf", "arial.ttf", "segoeui.ttf"])

    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT_UI_11 = load_font(11)
FONT_UI_13 = load_font(13)
FONT_UI_14 = load_font(14)
FONT_UI_15 = load_font(15)
FONT_UI_16 = load_font(16)
FONT_SERIF_18 = load_font(18, bold=True)
FONT_SERIF_26 = load_font(26, bold=True)
FONT_SERIF_28 = load_font(28, bold=True)
FONT_SERIF_30 = load_font(30, bold=True)
FONT_SERIF_34 = load_font(34, bold=True)


def rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], radius: int, fill, outline=None):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline)


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, font, fill):
    draw.text(xy, value, font=font, fill=fill)


def multiline(draw: ImageDraw.ImageDraw, xy: tuple[int, int], lines: list[str], font, fill, spacing: int = 8):
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        y += font.size + spacing


def wrap_text(draw: ImageDraw.ImageDraw, value: str, font, max_width: int) -> list[str]:
    words = value.split()
    if not words:
        return []

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    font,
    fill,
    max_width: int,
    spacing: int = 8,
) -> int:
    lines = wrap_text(draw, value, font=font, max_width=max_width)
    multiline(draw, xy, lines, font=font, fill=fill, spacing=spacing)
    line_height = font.size + spacing
    return max(0, len(lines) * line_height - spacing)


def gradient_background(size: tuple[int, int]) -> Image.Image:
    width, height = size
    image = Image.new("RGBA", size)
    pixels = image.load()
    for y in range(height):
        blend = y / max(height - 1, 1)
        r = int(BG_TOP[0] * (1 - blend) + BG_BOTTOM[0] * blend)
        g = int(BG_TOP[1] * (1 - blend) + BG_BOTTOM[1] * blend)
        b = int(BG_TOP[2] * (1 - blend) + BG_BOTTOM[2] * blend)
        for x in range(width):
            pixels[x, y] = (r, g, b, 255)
    return image


def add_orbs(base: Image.Image):
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((1120, -40, 1510, 350), fill=(211, 110, 67, 28))
    draw.ellipse((-80, 730, 320, 1130), fill=(31, 109, 98, 26))
    base.alpha_composite(overlay)


def desktop_mockup() -> Image.Image:
    img = gradient_background((2560, 1440))
    add_orbs(img)
    draw = ImageDraw.Draw(img)

    rounded(draw, (12, 12, 500, 1428), 36, PANEL, outline=(55, 41, 27, 20))
    rounded(draw, (530, 12, 2548, 308), 36, PANEL, outline=(55, 41, 27, 20))
    rounded(draw, (530, 338, 1810, 1428), 36, PANEL, outline=(55, 41, 27, 20))
    rounded(draw, (1840, 338, 2548, 1428), 36, PANEL, outline=(55, 41, 27, 20))

    rounded(draw, (44, 34, 120, 110), 24, ACCENT)
    text(draw, (72, 50), "H", FONT_SERIF_30, (255, 249, 241))
    text(draw, (144, 48), "SCIENTIFIC RAG", FONT_UI_11, ACCENT)
    text(draw, (144, 88), "Helio Retrieval", FONT_SERIF_34, INK)
    draw_wrapped_text(
        draw,
        (44, 148),
        "Scientific passage retrieval, reranking, and grounded question answering for OCR-processed literature collections.",
        FONT_UI_16,
        MUTED,
        max_width=390,
        spacing=6,
    )

    rounded(draw, (44, 238, 456, 650), 28, (255, 251, 245, 210), outline=(55, 41, 27, 18))
    text(draw, (78, 286), "Retrieval Controls", FONT_SERIF_26, INK)
    for label, value, y in [
        ("Top K", "4", 356),
        ("Vector Candidates", "12", 468),
        ("Keyword Candidates", "12", 580),
    ]:
        text(draw, (78, y), label, FONT_UI_16, MUTED)
        rounded(draw, (78, y + 26, 390, y + 82), 18, WHITE, outline=(55, 41, 27, 18))
        text(draw, (96, y + 39), value, FONT_UI_16, INK)

    rounded(draw, (44, 700, 456, 980), 28, (255, 251, 245, 210), outline=(55, 41, 27, 18))
    text(draw, (78, 748), "Research Prompts", FONT_SERIF_26, INK)
    for label, y, w in [
        ("eGFR performance", 806, 240),
        ("Renal monitoring limits", 874, 302),
        ("Cytokine combinations", 942, 286),
    ]:
        rounded(draw, (78, y, 78 + w, y + 48), 24, (31, 109, 98, 22))
        text(draw, (92, y + 14), label, FONT_UI_15, INK)

    rounded(draw, (44, 1252, 470, 1352), 28, (31, 109, 98, 28))
    draw.ellipse((84, 1298, 102, 1316), fill=GREEN)
    text(draw, (126, 1288), "PIPELINE", FONT_UI_11, ACCENT)
    text(draw, (126, 1322), "Ready for local requests", FONT_SERIF_18, INK)

    text(draw, (580, 54), "END-TO-END SCIENTIFIC RETRIEVAL", FONT_UI_11, ACCENT)
    draw_wrapped_text(
        draw,
        (580, 94),
        "Scientific document retrieval with traceable evidence",
        FONT_SERIF_34,
        INK,
        max_width=1300,
        spacing=8,
    )
    draw_wrapped_text(
        draw,
        (580, 154),
        "OCR, chunking, hybrid search, reranking, and cited answers are combined into one research workflow for scientific document retrieval.",
        FONT_UI_16,
        MUTED,
        max_width=1280,
        spacing=6,
    )

    chips = [
        ("Corpus", "OCR + Chunks", 580, 228, 290),
        ("Retrieval", "Hybrid Search", 900, 228, 290),
        ("Output", "Cited Answers", 1220, 228, 290),
    ]
    for label, value, x, y, w in chips:
        rounded(draw, (x, y, x + w, y + 48), 20, (255, 253, 248, 255), outline=(55, 41, 27, 18))
        text(draw, (x + 18, y + 15), label, FONT_UI_13, MUTED)
        text(draw, (x + 90, y + 12), value, FONT_SERIF_18, INK)

    text(draw, (580, 384), "SCIENTIFIC QA", FONT_UI_11, ACCENT)
    text(draw, (580, 430), "Literature Assistant", FONT_SERIF_34, INK)
    rounded(draw, (1502, 370, 1630, 422), 24, (36, 23, 14, 15))
    text(draw, (1546, 388), "Clear", FONT_UI_15, INK)

    rounded(draw, (580, 484, 632, 536), 18, WARM)
    text(draw, (596, 502), "AI", FONT_UI_15, (255, 249, 241))
    rounded(draw, (650, 484, 1414, 588), 24, PANEL_SOFT)
    draw_wrapped_text(
        draw,
        (690, 512),
        "Ask about methods, cohorts, biomarkers, limitations, or reported findings and I will answer from cited evidence.",
        FONT_UI_16,
        INK,
        max_width=660,
        spacing=6,
    )

    rounded(draw, (1040, 666, 1400, 782), 26, ACCENT)
    draw_wrapped_text(
        draw,
        (1082, 700),
        "How do the compared eGFR equations perform in AKI?",
        FONT_UI_16,
        (255, 249, 241),
        max_width=280,
        spacing=6,
    )
    rounded(draw, (1424, 702, 1478, 756), 18, WARM)
    text(draw, (1438, 720), "YOU", FONT_UI_13, (255, 249, 241))

    rounded(draw, (580, 842, 632, 894), 18, WARM)
    text(draw, (596, 860), "AI", FONT_UI_15, (255, 249, 241))
    rounded(draw, (650, 842, 1470, 1030), 24, PANEL_SOFT)
    draw_wrapped_text(
        draw,
        (690, 876),
        "The page compares multiple eGFR equations used for critically ill patients with AKI, emphasizing how well each tracks rapidly changing renal function and where their measurement limitations appear. [PMC3576793-leaf-002]",
        FONT_UI_16,
        INK,
        max_width=720,
        spacing=6,
    )

    text(draw, (580, 1218), "Research question", FONT_UI_14, MUTED)
    rounded(draw, (580, 1240, 1630, 1342), 24, WHITE, outline=(55, 41, 27, 18))
    draw_wrapped_text(
        draw,
        (622, 1274),
        "Ask about study design, biomarker results, limitations, or comparative findings...",
        FONT_UI_15,
        MUTED,
        max_width=600,
        spacing=6,
    )
    rounded(draw, (1316, 1258, 1630, 1342), 24, ACCENT_DARK)
    text(draw, (1424, 1292), "Run Retrieval", FONT_UI_15, (255, 249, 241))

    text(draw, (1888, 384), "EVIDENCE PANEL", FONT_UI_11, ACCENT)
    text(draw, (1888, 430), "Retrieved Passages", FONT_SERIF_34, INK)
    for y, title, subtitle, score, lines in [
        (
            502,
            "PMC3576793-leaf-002",
            "PMC3576793_00004 • Discussion",
            "hybrid 0.0152 • rerank 96",
            "The page compares eGFR equations in critically ill patients and discusses the challenge of measuring rapidly changing renal function.",
        ),
        (
            774,
            "PMC3576793-leaf-004",
            "PMC3576793_00005 • Results",
            "hybrid 0.0136 • rerank 88",
            "Additional context explains the assumptions and comparative performance of the equations.",
        ),
    ]:
        h = 204 if y == 502 else 188
        rounded(draw, (1888, y, 2380, y + h), 28, (255, 253, 248, 255), outline=(55, 41, 27, 18))
        text(draw, (1922, y + 34), title, FONT_UI_14, ACCENT)
        text(draw, (1922, y + 64), subtitle, FONT_UI_13, ACCENT)
        text(draw, (1922, y + 94), score, FONT_UI_13, ACCENT)
        draw_wrapped_text(
            draw,
            (1922, y + 136),
            lines,
            FONT_UI_14,
            MUTED,
            max_width=390,
            spacing=6,
        )

    return img.convert("RGB").resize((1600, 900), Image.Resampling.LANCZOS)


def mobile_mockup() -> Image.Image:
    img = gradient_background((720, 1440))
    add_orbs(img)
    draw = ImageDraw.Draw(img)

    rounded(draw, (24, 24, 696, 1416), 34, PANEL, outline=(55, 41, 27, 20))
    rounded(draw, (52, 52, 102, 102), 17, ACCENT)
    text(draw, (70, 64), "H", FONT_SERIF_26, (255, 249, 241))
    text(draw, (120, 72), "SCIENTIFIC RAG", FONT_UI_11, ACCENT)
    text(draw, (120, 96), "Helio Retrieval", FONT_SERIF_26, INK)

    rounded(draw, (52, 138, 668, 314), 26, (255, 253, 248, 255), outline=(55, 41, 27, 18))
    text(draw, (78, 166), "END-TO-END SCIENTIFIC RETRIEVAL", FONT_UI_11, ACCENT)
    text(draw, (78, 206), "Scientific retrieval on mobile", FONT_SERIF_28, INK)
    text(draw, (78, 242), "Review methods, findings, and evidence trails from indexed scientific pages.", FONT_UI_16, MUTED)
    for label, value, x, w in [
        ("Corpus", "OCR + Chunks", 78, 162),
        ("Search", "Hybrid", 254, 118),
        ("Output", "Cited Answers", 384, 158),
    ]:
        rounded(draw, (x, 266, x + w, 298), 16, (255, 253, 248, 255), outline=(55, 41, 27, 18))
        text(draw, (x + 12, 276), label, FONT_UI_13, MUTED)
        text(draw, (x + 62, 275), value, FONT_UI_14, INK)

    rounded(draw, (52, 336, 668, 474), 24, (255, 253, 248, 255), outline=(55, 41, 27, 18))
    text(draw, (78, 364), "Retrieval Controls", FONT_SERIF_18, INK)
    for label, x, w in [("Top K 4", 78, 120), ("Vector 12", 212, 180), ("Keyword 12", 406, 236)]:
        rounded(draw, (x, 392, x + w, 434), 15, WHITE, outline=(55, 41, 27, 18))
        text(draw, (x + 16, 406), label, FONT_UI_15, INK)

    text(draw, (52, 520), "SCIENTIFIC QA", FONT_UI_11, ACCENT)
    text(draw, (52, 556), "Literature Assistant", FONT_SERIF_28, INK)
    rounded(draw, (52, 590, 94, 632), 14, WARM)
    text(draw, (64, 602), "AI", FONT_UI_15, (255, 249, 241))
    rounded(draw, (106, 590, 622, 664), 18, PANEL_SOFT)
    multiline(draw, (128, 608), [
        "Ask about methods, cohorts, biomarkers, or findings",
        "and I'll answer from reranked cited evidence.",
    ], FONT_UI_16, INK, spacing=7)

    rounded(draw, (202, 694, 574, 776), 18, ACCENT)
    multiline(draw, (224, 714), [
        "What limitations are reported for tracking",
        "rapid renal function changes?",
    ], FONT_UI_15, (255, 249, 241), spacing=7)
    rounded(draw, (586, 714, 628, 756), 14, WARM)
    text(draw, (592, 728), "YOU", FONT_UI_13, (255, 249, 241))

    rounded(draw, (52, 804, 94, 846), 14, WARM)
    text(draw, (64, 816), "AI", FONT_UI_15, (255, 249, 241))
    rounded(draw, (106, 804, 638, 932), 18, PANEL_SOFT)
    multiline(draw, (128, 824), [
        "The retrieved passages say rapid renal function changes are",
        "difficult to capture accurately with static eGFR equations,",
        "especially in critically ill patients with AKI.",
        "[PMC3576793-leaf-002]",
    ], FONT_UI_15, INK, spacing=7)

    text(draw, (52, 982), "EVIDENCE PANEL", FONT_UI_11, ACCENT)
    text(draw, (52, 1018), "Retrieved Passages", FONT_SERIF_28, INK)
    rounded(draw, (52, 1048, 668, 1200), 24, (255, 253, 248, 255), outline=(55, 41, 27, 18))
    text(draw, (78, 1078), "PMC3576793-leaf-002", FONT_UI_14, ACCENT)
    text(draw, (78, 1102), "hybrid 0.0152 • rerank 96 • Discussion", FONT_UI_13, ACCENT)
    multiline(draw, (78, 1138), [
        "The page compares eGFR equations and the challenge of",
        "tracking rapidly changing renal function in AKI.",
    ], FONT_UI_15, MUTED, spacing=7)

    text(draw, (52, 1248), "Research question", FONT_UI_14, MUTED)
    rounded(draw, (52, 1266, 668, 1360), 22, WHITE, outline=(55, 41, 27, 18))
    text(draw, (78, 1308), "Ask about study design, biomarkers, limitations, or findings...", FONT_UI_15, MUTED)
    rounded(draw, (482, 1316, 668, 1360), 22, ACCENT_DARK)
    text(draw, (532, 1331), "Run Retrieval", FONT_UI_15, (255, 249, 241))

    return img.convert("RGB")


def main() -> int:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    desktop_mockup().save(DOCS_DIR / "chatbot_ui_desktop.png", format="PNG")
    mobile_mockup().save(DOCS_DIR / "chatbot_ui_mobile.png", format="PNG")
    print("Generated PNG mockups in docs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
