"""Non-destructive export QA and dark/light toolbar preview board."""
from pathlib import Path
from collections import deque
import json
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent.parent / 'assets' / 'mfo-toolbar-icons-astra'
NAMES = ['mfo-focus-surface', 'mfo-face-set', 'mfo-smart-face-set-fill', 'mfo-guided-ridge', 'mfo-tube-shape']
LABELS = ['Focus Surface', 'Face Set MFO', 'Smart Face Set Fill', 'Guided Ridge', 'Tube Shape']
board = Image.new('RGB', (1500, 700), '#202126')
draw = ImageDraw.Draw(board)
font_path = 'C:/Windows/Fonts/segoeui.ttf'
font = ImageFont.truetype(font_path, 20)
small = ImageFont.truetype(font_path, 16)
report = {}

for i, (name, label) in enumerate(zip(NAMES, LABELS)):
    img = Image.open(ASSETS / (name + '.png'))
    assert img.size == (512, 512) and img.mode == 'RGBA'
    a = img.getchannel('A')
    box = a.getbbox()
    histogram = a.histogram()
    assert box and min(box[0], box[1], 512-box[2], 512-box[3]) >= 12
    assert all(a.getpixel(p) == 0 for p in [(0,0), (511,0), (0,511), (511,511)])
    pixels = list(a.getdata())
    seen = bytearray(512*512)
    components = []
    for index, alpha in enumerate(pixels):
        if alpha < 32 or seen[index]:
            continue
        queue = deque([index])
        seen[index] = 1
        size = 0
        while queue:
            p = queue.popleft()
            size += 1
            x, y = p%512, p//512
            for xx, yy in ((x-1,y), (x+1,y), (x,y-1), (x,y+1)):
                if 0 <= xx < 512 and 0 <= yy < 512:
                    n = yy*512+xx
                    if pixels[n] >= 32 and not seen[n]:
                        seen[n] = 1
                        queue.append(n)
        components.append(size)
    report[name] = {'size': img.size, 'mode': img.mode, 'visible_bbox': box,
                    'transparent_pixels': histogram[0], 'opaque_pixels': histogram[255],
                    'antialiased_edge_pixels': sum(histogram[1:255]),
                    'components_alpha32': sorted(components, reverse=True)}
    x = 300*i
    draw.text((x+150, 17), label, fill='#E9E9EF', anchor='mt', font=font)
    thumb = img.resize((280, 280), Image.Resampling.LANCZOS)
    board.paste(thumb, (x+10, 54), thumb)
    draw.rectangle((x, 350, x+299, 595), fill='#E1E3E7')
    light_thumb = img.resize((230,230), Image.Resampling.LANCZOS)
    board.paste(light_thumb, (x+35, 355), light_thumb)
    draw.text((x+18, 620), '32 px', fill='#9A9DA7', font=small)
    icon32 = img.resize((32,32), Image.Resampling.LANCZOS)
    board.paste(icon32, (x+85, 614), icon32)
    draw.text((x+152, 620), '64 px', fill='#9A9DA7', font=small)
    icon64 = img.resize((64,64), Image.Resampling.LANCZOS)
    board.paste(icon64, (x+222, 601), icon64)

board.save(HERE / 'toolbar-preview.png')
(HERE / 'export-qa.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
