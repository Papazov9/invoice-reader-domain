"""PDF to text via Tesseract OCR (Bulgarian + English).

OCR dependencies are optional and import-guarded so the core service runs without a
system Tesseract install. Call ocr_status to check availability at runtime.

When OpenCV is installed the page images are deskewed, denoised and binarized before
recognition, which markedly improves accuracy on scanned invoices. Recognition also
returns a per-word confidence so callers can tell which tokens were uncertain and
fall back to a register or vision lookup for those fields.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from app.core import get_settings

try:  # optional heavy deps
    import pytesseract
    from pdf2image import convert_from_bytes, convert_from_path
    from pytesseract import Output

    _OCR_IMPORTED = True
except Exception:  # pragma: no cover - depends on environment
    _OCR_IMPORTED = False

try:  # image preprocessing is optional on top of OCR
    import cv2
    import numpy as np

    _CV2_IMPORTED = True
except Exception:  # pragma: no cover - depends on environment
    _CV2_IMPORTED = False


_TOKEN_STRIP = re.compile(r"[^0-9a-zA-Zа-яА-Я]+", re.UNICODE)


def normalize_token(text: str) -> str:
    """Lowercase a single token and strip surrounding punctuation/symbols."""
    return _TOKEN_STRIP.sub("", text or "").lower()


@dataclass
class OcrResult:
    """Recognised text plus the signals downstream extraction needs."""

    text: str = ""
    low_conf_tokens: set[str] = field(default_factory=set)
    mean_conf: float = 1.0  # 0..1; 1.0 when no confidence data is available
    page_images: list[bytes] = field(default_factory=list)  # PNG bytes per page


def ocr_status() -> dict[str, object]:
    """Report PDF-reading capability: a digital PDF needs only pdftotext (embedded text);
    a scanned one needs Tesseract."""
    settings = get_settings()
    has_pdftotext = shutil.which("pdftotext") is not None
    version = None
    if _OCR_IMPORTED:
        try:
            version = str(pytesseract.get_tesseract_version())
        except Exception:  # pragma: no cover - binary missing
            version = None
    tesseract_ok = version is not None
    status: dict[str, object] = {
        "available": tesseract_ok or has_pdftotext,
        "ocr": tesseract_ok,
        "pdftotext": has_pdftotext,
        "cv2": _CV2_IMPORTED,
        "preprocess": settings.ocr_preprocess and _CV2_IMPORTED,
        "vision": settings.ocr_vision_fallback and bool(settings.ocr_vision_model or settings.llm_model),
    }
    if version:
        status["tesseract_version"] = version
    if not status["available"]:
        status["reason"] = "no pdftotext and pytesseract/pdf2image not installed"
    elif not tesseract_ok:
        status["reason"] = "embedded-text only (no Tesseract; scanned PDFs unsupported)"
    return status


def _preprocess(image, photo: bool = False):
    """Deskew, denoise and binarize a page image to improve recognition.

    `photo=True` tunes for phone photos / scanned images: small ones are upscaled (tesseract
    needs adequate x-height) and an adaptive threshold is used so uneven lighting and shadows
    don't wipe out text the way a single global (Otsu) cutoff would. A pass-through when
    OpenCV is unavailable or preprocessing is disabled.
    """
    settings = get_settings()
    if not _CV2_IMPORTED or not settings.ocr_preprocess:
        return image

    from PIL import Image  # available whenever pdf2image is

    gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    if photo and max(gray.shape[:2]) < 2000:  # upscale low-res photos before OCR
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    if settings.ocr_denoise:
        gray = cv2.medianBlur(gray, 3)

    if settings.ocr_deskew:
        gray = _deskew(gray)

    mode = "adaptive" if photo else settings.ocr_threshold.lower()
    if mode == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive":
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
        )

    return Image.fromarray(gray)


def _deskew(gray):
    """Estimate the page skew from the text mask and rotate it upright."""
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) < 0.5:  # already straight; skip the warp
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )


def _ocr_data(image, lang: str) -> dict:
    try:
        return pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    except Exception:
        # Fall back to English if the bul language pack is missing.
        return pytesseract.image_to_data(image, lang="eng", output_type=Output.DICT)


def _lines_from_data(data: dict) -> list[list[tuple[int, str]]]:
    """Group recognised words into lines, in reading order. Each line is a list of
    (center_x, word)."""
    lines: dict[tuple, list[tuple[int, str]]] = {}
    order: list[tuple] = []
    n = len(data.get("text", []))
    for i in range(n):
        word = data["text"][i]
        if not word or not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        if key not in lines:
            lines[key] = []
            order.append(key)
        cx = int(data["left"][i]) + int(data["width"][i]) // 2
        lines[key].append((cx, word))
    return [sorted(lines[k], key=lambda w: w[0]) for k in order]


def _is_two_column(line: list[tuple[int, str]], width: int) -> bool:
    """A line straddles a central gutter: it has words clearly left of 45% and right
    of 55% of the page width, with the middle empty."""
    return (
        any(cx < width * 0.45 for cx, _ in line)
        and any(cx > width * 0.55 for cx, _ in line)
    )


def _image_to_text(data: dict, width: int) -> str:
    """Reflow words into text. A two-column header (supplier beside recipient, common on
    Bulgarian invoices) is read top-to-bottom per column instead of interleaved across
    the gutter, so the two parties don't bleed into each other. Only the leading band of
    two-column lines is reflowed; the rest of the page stays in reading order, so totals
    and tables are untouched."""
    lines = _lines_from_data(data)
    mid = width / 2

    band = 0
    while band < len(lines) and _is_two_column(lines[band], width):
        band += 1

    out: list[str] = []
    if band >= 2:  # a real column region, not one stray straddling line
        left = [" ".join(w for cx, w in ln if cx <= mid) for ln in lines[:band]]
        right = [" ".join(w for cx, w in ln if cx > mid) for ln in lines[:band]]
        out.append("\n".join(s for s in left if s.strip()))
        out.append("\n".join(s for s in right if s.strip()))
        rest = lines[band:]
    else:
        rest = lines

    out.extend(" ".join(w for _, w in ln) for ln in rest)
    return "\n".join(out)


def _collect_confidence(data: dict, low: set[str], confs: list[float], threshold: float) -> None:
    n = len(data.get("text", []))
    for i in range(n):
        word = data["text"][i]
        if not word or not word.strip():
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        conf /= 100.0
        confs.append(conf)
        if conf < threshold:
            token = normalize_token(word)
            if token:
                low.add(token)


def _images_to_result(images, photo: bool = False) -> OcrResult:
    settings = get_settings()
    lang = settings.ocr_languages
    threshold = settings.ocr_word_conf_min
    parts: list[str] = []
    low: set[str] = set()
    confs: list[float] = []
    pages: list[bytes] = []
    for image in images:
        prepared = _preprocess(image, photo=photo)
        data = _ocr_data(prepared, lang)
        parts.append(_image_to_text(data, prepared.width))
        _collect_confidence(data, low, confs, threshold)
        # Vision fallback gets the ORIGINAL page, not the binarized one: Tesseract wants a
        # thresholded image, but a vision model reads the natural colour/greyscale page more
        # accurately (adaptive thresholding drops the faint strokes it relies on).
        pages.append(_to_png(image))
    mean_conf = sum(confs) / len(confs) if confs else 1.0
    return OcrResult(
        text="\n\n".join(parts),
        low_conf_tokens=low,
        mean_conf=mean_conf,
        page_images=pages,
    )


def _to_png(image) -> bytes:
    buf = io.BytesIO()
    image.convert("L").save(buf, format="PNG")
    return buf.getvalue()


def _looks_like_text(text: str) -> bool:
    """Whether the embedded layer is usable. It must carry real Bulgarian/Latin letters:
    a scanned PDF yields almost none, and some PDFs have a broken font encoding whose
    text extracts as mojibake (exotic Latin-Extended/IPA code points that ARE letters but
    not real script) — both are rejected so we OCR the rendered page instead."""
    real = sum(
        1 for c in text
        if ("Ѐ" <= c <= "ӿ") or ("a" <= c <= "z") or ("A" <= c <= "Z")
    )
    # A broken font encoding renders Cyrillic as exotic Latin-Extended/IPA glyphs (e.g.
    # "ǻȇȑșȚȗȇȶ" for "Фактура") that ARE letters but not real script. When they dominate,
    # the embedded layer is useless even if Latin company names push `real` past 200 — OCR
    # the rendered page instead.
    exotic = sum(1 for c in text if ("ƀ" <= c <= "ʯ") or ("ᴀ" <= c <= "ỿ"))
    if exotic > 80 and exotic > real * 0.5:
        return False
    return real >= 200


def _embedded_text(path: str) -> str | None:
    """Extract a PDF's embedded text layer via poppler's pdftotext, or None when the
    PDF is image-only / pdftotext is unavailable. Exact and fast, so it is preferred
    over OCR for digital PDFs."""
    if not shutil.which("pdftotext"):
        return None
    try:
        out = subprocess.run(
            ["pdftotext", "-layout", "-enc", "UTF-8", path, "-"],
            capture_output=True, timeout=60,
        )
    except Exception:  # pragma: no cover - depends on environment
        return None
    if out.returncode != 0:
        return None
    text = out.stdout.decode("utf-8", "replace")
    return text if _looks_like_text(text) else None


def _embedded_result(path: str) -> OcrResult | None:
    if not get_settings().ocr_prefer_embedded_text:
        return None
    text = _embedded_text(path)
    if not text:
        return None
    # Exact text: full confidence, no uncertain tokens, no need for the vision fallback.
    return OcrResult(text=text, low_conf_tokens=set(), mean_conf=1.0, page_images=[])


def extract_ocr_from_pdf(path: str) -> OcrResult:
    embedded = _embedded_result(path)
    if embedded is not None:
        return embedded
    if not _OCR_IMPORTED:
        raise RuntimeError("OCR dependencies not available (see requirements.txt)")
    images = convert_from_path(path, dpi=get_settings().ocr_dpi)
    return _images_to_result(images)


def extract_ocr_from_pdf_bytes(content: bytes) -> OcrResult:
    if get_settings().ocr_prefer_embedded_text and shutil.which("pdftotext"):
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(content)
                tmp = f.name
            embedded = _embedded_result(tmp)
            if embedded is not None:
                return embedded
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:  # pragma: no cover
                    pass
    if not _OCR_IMPORTED:
        raise RuntimeError("OCR dependencies not available (see requirements.txt)")
    images = convert_from_bytes(content, dpi=get_settings().ocr_dpi)
    return _images_to_result(images)


def render_pdf_page_images(content: bytes, max_pages: int = 4) -> list[bytes]:
    """Render the first pages of a PDF to PNG bytes for the vision model. Used only as a
    lazy fallback when a digital PDF's embedded text was read (so no page images were made)
    but the rule extraction came out critically incomplete. Empty list if rendering is
    unavailable or fails."""
    if not _OCR_IMPORTED:
        return []
    try:
        images = convert_from_bytes(content, dpi=get_settings().ocr_dpi)
    except Exception:  # pragma: no cover - depends on environment
        return []
    return [_to_png(img) for img in images[:max_pages]]


def extract_ocr_from_image_bytes(content: bytes) -> OcrResult:
    """OCR a photographed/scanned image (JPG/PNG/TIFF…) through the same preprocess +
    column-reflow pipeline as PDF pages. EXIF orientation is honoured so phone photos are
    upright, and multi-frame images (e.g. TIFF) are read page by page."""
    if not _OCR_IMPORTED:
        raise RuntimeError("OCR dependencies not available (see requirements.txt)")
    from PIL import Image, ImageOps, ImageSequence

    img = Image.open(io.BytesIO(content))
    frames = [ImageOps.exif_transpose(frame.convert("RGB")) for frame in ImageSequence.Iterator(img)]
    return _images_to_result(frames or [img.convert("RGB")], photo=True)


def extract_text_from_pdf(path: str) -> str:
    """Recognise a PDF and return just the text (back-compatible helper)."""
    return extract_ocr_from_pdf(path).text


def extract_text_from_pdf_bytes(content: bytes) -> str:
    """Recognise a PDF from bytes and return just the text (back-compatible helper)."""
    return extract_ocr_from_pdf_bytes(content).text
