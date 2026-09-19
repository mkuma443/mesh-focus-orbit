# MFO toolbar icons — Astra Blender source

The authoritative reference is `approved-reference.png` in this directory. All five icons are constructed as real Blender geometry; the reference is not used as a texture or cut out to produce the PNGs.

## Deliverables

Five individual, transparent 512 × 512 RGBA PNGs are exported to `../../assets/mfo-toolbar-icons-astra/` with the existing descriptive feature filenames. `mfo-toolbar-icons-astra.blend` contains one named scene per icon. The camera, lighting, meshes, ribbon bevels and materials remain editable.

`render_astra_icons.py` is the deterministic source. Rebuild all five scenes and save the `.blend` with:

```powershell
& 'E:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe' --background --factory-startup --python 'work\astra-icon-modeling\render_astra_icons.py'
```

Append `-- focus face_set fill ridge tube` with any subset to render only those icons. A subset run does not replace the complete five-scene `.blend`.

`generate_toolbar_icons_dat.py` temporarily evaluates each scene with that
scene's own `ViewLayer`/depsgraph, then restores the caller's context. This is
important for the Bevel/Solidify modifiers on the Face Set, Fill, Guided Ridge,
and Tube assets; the report printed by the generator includes source and
evaluated mesh counts as a regression check. Pass `--report
work/astra-icon-modeling/generator-report.json` to retain that report.

## Authored details

- Focus Surface uses a compact angular charcoal mass, a horizontal beveled lavender orbit ribbon, a readable arrowhead and a glossy purple surface seed.
- Face Set MFO shares the faceted form, colors a connected group of faces purple, and exposes the orbit cue on the right side so that the selected region stays visible.
- Smart Face Set Fill uses a slightly curved mesh patch, a gray boundary margin, connected green/yellow/red regions and a white seed.
- Guided Ridge is a closed solid mound. Its lavender rail lies on a curved crest and follows the descending surface to the front edge.
- Tube Shape is a bent, near-uniform bundle with a visible triangulated end profile. Two beveled bands follow local cross-section planes perpendicular to the centerline.

The shared visual language uses broad pale mesh edges, matte charcoal faces, lavender bevels, orthographic cameras and directional studio lighting. There are no image textures, glow, environment background, ground plane or neighboring objects in an export.

## Verification

`inspect_icons.py` uses Pillow to create `toolbar-preview.png` on dark and light backgrounds with actual 32 px and 64 px thumbnails. It also writes `export-qa.json`, checking dimensions, RGBA mode, fully transparent canvas corners, safe canvas margins and connected alpha regions. It never modifies the exported icons.

```powershell
& 'C:\Users\mkuma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'work\astra-icon-modeling\inspect_icons.py'
```

Existing concept assets and add-on source are preserved. Only this source directory and the separate Astra output directory are used for generated deliverables.

For QA of the actual VCO files (rather than the reference PNGs), run:

```powershell
& 'C:\Users\mkuma\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' 'work\astra-icon-modeling\rasterize_vco_icons.py' `
  --input-dir 'assets\mfo-toolbar-icons-astra' `
  --output 'work\astra-icon-modeling\vco-toolbar-preview.png' `
  --report 'work\astra-icon-modeling\vco-qa.json' `
  --png-dir 'assets\mfo-toolbar-icons-astra'
```

The QA image rasterizes the triangles and RGBA payload directly from each
`.dat` at both 32 px and 64 px on dark and light backgrounds (enlarged with
nearest-neighbor only for the composite); the JSON also records source PNG
alpha bounds for a basic coverage comparison.
