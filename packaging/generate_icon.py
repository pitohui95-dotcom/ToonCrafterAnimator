"""Generate icon.png / icon.ico without requiring design software."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src" / "tooncrafter_animator" / "assets"


def _pixel(x: int, y: int, size: int) -> tuple[int, int, int]:
    # Dark field, indigo disc, gold frame marks — reads as "in-between frames".
    cx, cy = size / 2, size / 2
    dx, dy = x - cx, y - cy
    r2 = dx * dx + dy * dy
    bg = (18, 18, 24)
    if r2 > (size * 0.46) ** 2:
        return bg
    # two keyframe bars
    if abs(dx) > size * 0.28 and abs(dy) < size * 0.18:
        return (242, 242, 242)
    # in-between dots
    if abs(dy) < size * 0.07 and abs(dx) < size * 0.22:
        return (138, 180, 255)
    ring = abs(r2 ** 0.5 - size * 0.38)
    if ring < size * 0.035:
        return (255, 224, 130)
    return (28, 32, 48)


def write_png(path: Path, size: int = 256) -> None:
    raw = bytearray()
    for y in range(size):
        raw.append(0)
        for x in range(size):
            r, g, b = _pixel(x, y, size)
            raw.extend((r, g, b))
    compressor = zlib.compressobj(9, zlib.DEFLATED, zlib.MAX_WBITS)
    compressed = compressor.compress(bytes(raw)) + compressor.flush()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_ico(png_path: Path, ico_path: Path) -> None:
    try:
        from PIL import Image
    except ImportError:
        # Minimal ICO wrapping a PNG (Vista+).
        png = png_path.read_bytes()
        # ICONDIR + one ICONDIRENTRY + PNG
        header = struct.pack("<HHH", 0, 1, 1)
        entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(png), 22)
        ico_path.write_bytes(header + entry + png)
        return
    img = Image.open(png_path).convert("RGBA")
    img.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (256, 256)])


def main() -> None:
    png = ASSETS / "icon.png"
    ico = ASSETS / "icon.ico"
    write_png(png, 256)
    write_ico(png, ico)
    print(f"wrote {png} {ico}")


if __name__ == "__main__":
    main()
