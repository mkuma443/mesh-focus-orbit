"""Rasterize Blender VCO toolbar icons for a small, deterministic visual QA.

The input is the actual ``.dat`` payload consumed by
``bpy.app.icons.new_triangles_from_file``.  This is intentionally separate
from the PNG inspection: it catches malformed coordinates, color payloads,
empty triangles, and clipping introduced during VCO export.

Example (from the repository root)::

    python work/astra-icon-modeling/rasterize_vco_icons.py \
        --input-dir assets/mfo-toolbar-icons-astra \
        --output work/astra-icon-modeling/vco-toolbar-preview.png \
        --report work/astra-icon-modeling/vco-qa.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct

from PIL import Image, ImageDraw


STEMS = (
    "mfo-focus-surface",
    "mfo-face-set",
    "mfo-smart-face-set-fill",
    "mfo-guided-ridge",
    "mfo-tube-shape",
)


def read_vco(path):
    payload = pathlib.Path(path).read_bytes()
    if payload[:8] != b"VCO\x00\xff\xff\x00\x00":
        raise ValueError(f"invalid VCO header: {path}")
    body = payload[8:]
    if len(body) % 18:
        raise ValueError(f"invalid VCO payload length: {path}")
    triangle_count = len(body) // 18
    coords = body[: triangle_count * 6]
    colors = body[triangle_count * 6 :]
    triangles = []
    for index in range(triangle_count):
        points = tuple(
            (coords[index * 6 + offset], coords[index * 6 + offset + 1])
            for offset in (0, 2, 4)
        )
        rgba = tuple(
            tuple(colors[index * 12 + offset : index * 12 + offset + 4])
            for offset in (0, 4, 8)
        )
        triangles.append((points, rgba))
    return triangles


def rasterize(triangles, size, background):
    image = Image.new("RGBA", (size, size), tuple(background) + (255,))
    draw = ImageDraw.Draw(image, "RGBA")
    for points, colors in triangles:
        # VCO stores one color per vertex. A flat average is sufficient for
        # QA and preserves the expected feature colors without inventing a
        # shader or depending on Blender's UI renderer.
        color = tuple(sum(vertex[channel] for vertex in colors) // 3 for channel in range(4))
        scaled = tuple((int(x) * (size - 1) / 255.0, int(y) * (size - 1) / 255.0) for x, y in points)
        draw.polygon(scaled, fill=color)
    return image


def alpha_bbox(image, background):
    pixels = image.convert("RGB")
    mask = Image.new("L", image.size, 0)
    source = pixels.load()
    output = mask.load()
    for y in range(image.height):
        for x in range(image.width):
            output[x, y] = 255 if source[x, y] != tuple(background) else 0
    return mask.getbbox(), sum(value > 0 for value in mask.tobytes())


def source_png_metrics(path):
    image = Image.open(path).convert("RGBA")
    alpha = image.getchannel("A")
    return {
        "size": list(image.size),
        "alpha_bbox": list(alpha.getbbox()) if alpha.getbbox() else None,
        "alpha_pixels": sum(value > 0 for value in alpha.tobytes()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--png-dir")
    args = parser.parse_args(argv)
    input_dir = pathlib.Path(args.input_dir)
    output_path = pathlib.Path(args.output)
    report_path = pathlib.Path(args.report)
    backgrounds = ((32, 33, 38), (224, 226, 230))
    sizes = (32, 64)
    cell_width = 190
    cell_height = 145
    image = Image.new(
        "RGB",
        (cell_width * len(STEMS), cell_height * len(backgrounds) * len(sizes)),
        backgrounds[0],
    )
    report = {}
    for background_index, background in enumerate(backgrounds):
        for size_index, size in enumerate(sizes):
            for column, stem in enumerate(STEMS):
                triangles = read_vco(input_dir / f"{stem}.dat")
                thumb = rasterize(triangles, size, background).convert("RGB")
                display = thumb.resize((128, 128), Image.Resampling.NEAREST)
                x = column * cell_width + (cell_width - 128) // 2
                y = (background_index * len(sizes) + size_index) * cell_height + 8
                image.paste(display, (x, y))
                bbox, coverage = alpha_bbox(thumb, background)
                report.setdefault(stem, {"triangles": len(triangles), "renders": []})[
                    "renders"
                ].append({
                    "background": list(background),
                    "size": size,
                    "bbox": list(bbox) if bbox else None,
                    "coverage": coverage,
                })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    if args.png_dir:
        png_dir = pathlib.Path(args.png_dir)
        for stem in STEMS:
            source = png_dir / f"{stem}.png"
            if source.is_file():
                report[stem]["source_png"] = source_png_metrics(source)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
