"""Utilidades para grabar videos de episodios con texto sobreimpreso."""

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.load_default(size=18)
    return _FONT


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Dibuja lineas de texto sobre una franja semitransparente en la parte superior."""
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, img.width, 10 + 24 * len(lines)], fill=(15, 20, 30, 170))
    for i, line in enumerate(lines):
        draw.text((10, 6 + 24 * i), line, fill=(240, 240, 240, 255), font=_font())
    return np.asarray(Image.alpha_composite(img, overlay).convert("RGB"))


def write_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    """Guarda los frames como .mp4 (H.264) o .gif segun la extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gif":
        imageio.mimsave(path, frames, duration=1000 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=7, macro_block_size=8)
    return path
