"""Astra's native VCO icon for Select Back Faces.

Regenerate with Python (Pillow is needed only for preview PNGs).
The editable geometry below uses a top-left 255-unit design canvas.
The exporter flips Y and fixes winding for Blender's VCO triangle format.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[2]
STEM = "mfo-select-back-faces"
OUT = ROOT / "assets" / "mfo-toolbar-icons-astra"


def artwork():
    shapes = []

    def polygon(points, color):
        shapes.append((points, color))

    def line(a, b, width, color):
        dx, dy = b[0] - a[0], b[1] - a[1]
        scale = width / (2 * math.hypot(dx, dy))
        nx, ny = -dy * scale, dx * scale
        polygon([(a[0]+nx, a[1]+ny), (b[0]+nx, b[1]+ny),
                 (b[0]-nx, b[1]-ny), (a[0]-nx, a[1]-ny)], color)

    def outline(points, width, color, closed=True):
        for a, b in zip(points, points[1:] + (points[:1] if closed else [])):
            line(a, b, width, color)

    # Angular surrounding mesh: subdued gray matches the five Astra tools.
    a, b, c = (39, 64), (113, 31), (204, 68)
    d, e, f = (201, 163), (112, 213), (26, 170)
    g, h, i = (99, 106), (148, 129), (97, 170)
    for points, color in [
        ([a, b, g], "#92979f"), ([b, c, g], "#797f89"),
        ([a, g, f], "#666c77"), ([f, g, i], "#555b65"),
        ([f, i, e], "#404650"), ([i, d, e], "#505762"),
        ([c, d, h], "#69707b"),
        # A dark irregular opening remains visible under the lifted face.
        ([g, c, h], "#242832"), ([g, h, i], "#242832"),
        ([h, d, i], "#303540"),
    ]:
        polygon(points, color)
    outline([a, b, c, d, e, f], 3.4, "#d4d9e0")
    for p, q in [(a, g), (b, g), (f, g), (f, i), (i, e), (i, d)]:
        line(p, q, 2.4, "#b1b7c1")

    # Folded back-facing patch: two orange triangles and a pale fold lip.
    tip, ridge, base = (161, 50), (206, 100), (156, 160)
    polygon([g, tip, ridge], "#ffb34f")
    polygon([g, ridge, base], "#ed792e")
    outline([g, tip, ridge, base], 3.8, "#ffe0a4")
    line(g, ridge, 2.4, "#ffd087")
    polygon([g, (119, 90), tip], "#fff0c4")

    # Small white pointer is independent of the mesh silhouette.
    # Use convex parts rather than a concave fan so VCO stays well formed.
    cursor = [(171, 157), (226, 188), (203, 194), (192, 216)]
    polygon(cursor, "#242832")
    outline(cursor, 8.0, "#242832")
    polygon([(175, 164), (218, 188), (199, 190), (191, 207)], "#f5f7fa")
    return shapes


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    coords, colors = bytearray(), bytearray()
    triangles = []
    shapes = artwork()
    for points, color in shapes:
        rgba = tuple(bytes.fromhex(color[1:])) + (255,)
        for n in range(1, len(points)-1):
            triangle = [points[0], points[n], points[n+1]]
            encoded = [(round(x), 255-round(y)) for x, y in triangle]
            cross = ((encoded[1][0]-encoded[0][0]) * (encoded[2][1]-encoded[0][1])
                     - (encoded[1][1]-encoded[0][1]) * (encoded[2][0]-encoded[0][0]))
            if cross == 0:
                continue
            if cross < 0:
                encoded[1], encoded[2] = encoded[2], encoded[1]
            assert all(0 < v < 255 for p in encoded for v in p)
            coords.extend(v for p in encoded for v in p)
            colors.extend(rgba * 3)
            triangles.append(([(x, 255-y) for x, y in encoded], rgba))
    payload = b"VCO\x00\xff\xff\x00\x00" + coords + colors
    assert (len(payload)-8) % 18 == 0
    (OUT / (STEM + ".dat")).write_bytes(payload)
    polygons = "\n".join(
        '<polygon points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        + f'" fill="{color}"/>' for points, color in shapes
    )
    (OUT / (STEM + ".svg")).write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 255 255">'
        '<title>Select Back Faces: lifted orange mesh faces and selection pointer</title>'
        + polygons + "</svg>\n", encoding="utf-8")
    # Preview comes from the exported/quantized triangles, not a second design.
    from PIL import Image, ImageDraw
    scale = 4
    im = Image.new("RGBA", (512*scale, 512*scale))
    draw = ImageDraw.Draw(im)
    for points, rgba in triangles:
        draw.polygon([(x*512*scale/255, y*512*scale/255) for x,y in points], fill=rgba)
    im = im.resize((512, 512), Image.Resampling.LANCZOS)
    im.save(OUT / (STEM + ".png"))
    qa = Image.new("RGB", (600, 180), "#202126")
    qdraw = ImageDraw.Draw(qa)
    for n, size in enumerate((24, 32, 48, 64)):
        thumb = im.resize((size, size), Image.Resampling.LANCZOS)
        x = 30 + n*140
        qa.paste(thumb, (x, 45), thumb)
        qdraw.text((x, 125), str(size) + " px", fill="white")
    qa.save(Path(__file__).with_name("backface-icon-sizes.png"))
    print(json.dumps({"triangles": len(triangles), "bytes": len(payload),
                      "icon": str(OUT / (STEM + ".dat")), "sizes": [24,32,48,64]}))


if __name__ == "__main__":
    export()
