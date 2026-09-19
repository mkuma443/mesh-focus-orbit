"""Authored Blender geometry for the five Mesh Focus Orbit toolbar icons.

Run with Blender 5.2.2: blender --background --factory-startup --python <this file>
Optional: -- focus face_set fill ridge tube (render only named concepts).
All geometry, materials, lighting, cameras and transparent renders are reproducible.
No external asset, ImageGen, SVG, add-on source or active Blender session is used.
"""
import bpy
import math
import os
import sys
from mathutils import Vector, Euler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, '..', '..', 'assets', 'mfo-toolbar-icons-astra'))
os.makedirs(OUT, exist_ok=True)
TAU = math.tau


def srgb(hex_color):
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in rgb) + (1,)


def mat(name, color, rough=.72, emission=.13):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    m.diffuse_color = srgb(color)
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = m.diffuse_color
    p.inputs['Roughness'].default_value = rough
    p.inputs['Specular IOR Level'].default_value = .22
    p.inputs['Emission Color'].default_value = m.diffuse_color
    p.inputs['Emission Strength'].default_value = emission
    return m


def materials():
    return {
        'clay': mat('MFO | cool sculpt clay', '696B73', emission=.025),
        'clayLight': mat('MFO | upper facets', '777A84', emission=.045),
        'clayDark': mat('MFO | underside', '474B55', emission=.045),
        'edge': mat('MFO | mesh seams', 'C9CCD5', emission=.20),
        'purple': mat('MFO | purple active surface', '965DDC', emission=.12),
        'purpleLight': mat('MFO | purple controls', 'AD83EE', rough=.32, emission=.10),
        'purpleDark': mat('MFO | purple region boundary', '6237AC'),
        'purpleSide': mat('MFO | lavender ribbon sides', '8458CD', rough=.38, emission=.10),
        'purpleEdge': mat('MFO | lavender bevel glint', 'CAAAFB', rough=.30, emission=.15),
        'green': mat('MFO | face set green', '7EAF53'),
        'yellow': mat('MFO | face set yellow', 'E5B947'),
        'red': mat('MFO | face set red', 'DE6560'),
        'white': mat('MFO | focus indicator', 'F4F5FF', emission=.35),
    }


M = materials()


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def scene(name, scale=4.05, camera=(0, -10, 0), target=(0, 0, 0)):
    s = bpy.data.scenes.new(name)
    bpy.context.window.scene = s
    s.render.engine = 'BLENDER_EEVEE'
    s.render.resolution_x = 512
    s.render.resolution_y = 512
    s.render.resolution_percentage = 100
    s.render.image_settings.file_format = 'PNG'
    s.render.image_settings.color_mode = 'RGBA'
    s.render.image_settings.color_depth = '8'
    s.render.film_transparent = True
    s.render.dither_intensity = 0.0
    s.render.image_settings.compression = 35
    if hasattr(s.eevee, 'taa_render_samples'):
        s.eevee.taa_render_samples = 128
    s.view_settings.view_transform = 'Standard'
    s.view_settings.look = 'None'
    s.view_settings.exposure = 0
    s.view_settings.gamma = 1
    s.world = bpy.data.worlds.new(name + ' | ambient')
    s.world.use_nodes = True
    s.world.node_tree.nodes['Background'].inputs[0].default_value = (.25, .25, .25, 1)
    s.world.node_tree.nodes['Background'].inputs[1].default_value = .15
    bpy.ops.object.camera_add(location=camera)
    s.camera = bpy.context.object
    s.camera.name = name + ' | orthographic camera'
    s.camera.data.type = 'ORTHO'
    s.camera.data.ortho_scale = scale
    look_at(s.camera, target)
    for label, loc, power, size in [
        ('large key', (-3.5, -4.5, 6), 850, 3.2),
        ('soft fill', (4, -3, 1.6), 40, 5),
        ('edge light', (.5, 3, 4), 220, 4),
    ]:
        bpy.ops.object.light_add(type='AREA', location=loc)
        lamp = bpy.context.object
        lamp.name = name + ' | ' + label
        lamp.data.energy = power
        lamp.data.shape = 'DISK'
        lamp.data.size = size
        look_at(lamp, (0, 0, 0))
    s['design_note'] = '512px RGBA, transparent film; modeled shapes and physical lighting, no bitmap textures.'
    return s


def mesh(name, verts, faces, mats, indices=None):
    data = bpy.data.meshes.new(name)
    data.from_pydata(verts, [], faces)
    data.update()
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    for material in mats:
        data.materials.append(material)
    if indices:
        for polygon, index in zip(data.polygons, indices):
            polygon.material_index = index
    return obj


def curve(name, points, radius, material, cyclic=False, smooth=True):
    data = bpy.data.curves.new(name, 'CURVE')
    data.dimensions = '3D'
    data.bevel_depth = radius
    data.bevel_resolution = 3
    data.resolution_u = 1
    data.use_fill_caps = True
    spline = data.splines.new('POLY')
    spline.points.add(len(points) - 1)
    for p, xyz in zip(spline.points, points):
        p.co = (*xyz, 1)
    spline.use_cyclic_u = cyclic
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(material)
    return obj


def ball(name, position, radius, material, subdivision=2):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivision, radius=radius, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def glossy_ball(name, position, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def mesh_edges(obj, radius=.009):
    for i, edge in enumerate(obj.data.edges):
        curve('Topology | ' + obj.name + ' %03d' % i,
              [obj.matrix_world @ obj.data.vertices[j].co for j in edge.vertices], radius, M['edge'])


def ribbon(name, centers, sideways, width, thickness=.06):
    verts, faces, indices = [], [], []
    for i, (center, side) in enumerate(zip(centers, sideways)):
        center, side = Vector(center), Vector(side).normalized()
        tangent = (Vector(centers[min(len(centers)-1, i+1)]) - Vector(centers[max(0, i-1)])).normalized()
        normal = tangent.cross(side).normalized()
        for side_sign, height_sign in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            verts.append(center + side * side_sign * width / 2 + normal * height_sign * thickness / 2)
    for i in range(len(centers)-1):
        for j in range(4):
            a = 4*i+j
            b = 4*i+(j+1)%4
            faces.append((a, b, b+4, a+4))
            indices.append(0 if j in (0, 2) else 1)
    faces.extend([(3, 2, 1, 0), tuple(4*(len(centers)-1)+j for j in range(4))])
    indices.extend((1, 1))
    obj = mesh(name, verts, faces, [M['purpleLight'], M['purpleSide'], M['purpleEdge']], indices)
    bevel = obj.modifiers.new('Fine rounded ribbon bevel', 'BEVEL')
    bevel.width = .013
    bevel.segments = 3
    bevel.affect = 'EDGES'
    bevel.material = 2
    bevel.limit_method = 'ANGLE'
    bevel.angle_limit = .35
    normal = obj.modifiers.new('Weighted ribbon normals', 'WEIGHTED_NORMAL')
    normal.keep_sharp = True
    return obj


def region_boundary(obj, material, radius=.013):
    edge_mats = {}
    for polygon in obj.data.polygons:
        for edge in polygon.edge_keys:
            edge_mats.setdefault(tuple(sorted(edge)), set()).add(polygon.material_index)
    for i, (edge, indices) in enumerate(edge_mats.items()):
        if len(indices) > 1:
            points = [obj.matrix_world @ (obj.data.vertices[j].co * 1.005) for j in edge]
            curve('Face-set boundary %02d' % i, points, radius, material)


def sculpt_form(name, region=False, fill=False):
    verts = [(0.02, .02, 1.32)]
    sides = 7
    for level, (radius, height, turn) in enumerate(((.83, .82, .12), (1.06, .05, -.10), (.86, -.78, .14))):
        for j in range(sides):
            angle = TAU*j/sides + .25 + turn
            verts.append((radius*math.cos(angle), .87*radius*math.sin(angle), height + .035*math.sin(angle*2)))
    verts.append((-.13, .02, -1.29))
    faces = [(0, 1+j, 1+(j+1)%sides) for j in range(sides)]
    for level in range(2):
        for j in range(sides):
            a = 1 + level*sides+j
            b = 1 + level*sides+(j+1)%sides
            c = a+sides
            d = b+sides
            faces.extend(((a, c, b), (b, c, d)))
    faces.extend((15+j, 22, 15+(j+1)%sides) for j in range(sides))
    obj = mesh(name, verts, faces, [M['clay'], M['purple']])
    for polygon in obj.data.polygons:
        c = polygon.center
        polygon.material_index = int(region and c.x > -.10 and -.54 < c.z < .80 and c.y < .10)
    mesh_edges(obj, .010)
    return obj


def orbit(face_set=False):
    if face_set:
        angles = [1.95 - 2.83*i/96 for i in range(97)]
        centers = [Vector((.57 + .90*math.cos(a), -.34 + .26*math.sin(a), .19 + .80*math.sin(a))) for a in angles]
        sideways = [(math.cos(a), 0, math.sin(a)) for a in angles]
        ribbon('Orbit | Face Set curved lavender ribbon', centers, sideways, .175, .085)
        c = centers[-1]
        c.y = -.93
        tip = c + Vector((-.15, -.035, -.26))
        left = c + Vector((-.26, -.035, .16))
        right = c + Vector((.18, -.035, .04))
        verts = [p + Vector((0, d, 0)) for d in (-.06, .06) for p in (tip, left, right)]
        obj = mesh('Orbit | Face Set arrowhead', verts, [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)], [M['purpleLight'], M['purpleSide'], M['purpleEdge']], [0,0,1,1,1])
        bevel = obj.modifiers.new('Arrow bevel', 'BEVEL')
        bevel.width = .018
        bevel.segments = 3
        bevel.material = 2
        obj.modifiers.new('Weighted arrow normals', 'WEIGHTED_NORMAL')
        return
    u = Vector((1, 0, 0))
    v = Vector((0, .932, .362)).normalized()
    c = Vector((0, 0, -.040))
    r = 1.59
    start, end = 0, TAU
    pts = [c + r * (math.cos(t) * u + math.sin(t) * v) for t in [start + (end - start) * i / 128 for i in range(129)]]
    sideways = [(point-c).normalized() for point in pts]
    ribbon('Orbit | broad beveled lavender ribbon', pts, sideways, .195, .090)
    # The terminal arrow is pitched toward the viewer to retain its triangle at 32px.
    tip = Vector((1.78, -.93, -.205))
    upper = Vector((1.29, -.93, -.11))
    lower = Vector((1.43, -.93, -.69))
    normal = Vector((0, 1, 0))
    verts = [tip + normal * h for h in (-.056, .056)]
    for h in (-.056, .056):
        verts.extend([upper + normal*h, lower + normal*h])
    obj = mesh('Orbit | broad directional arrowhead', verts, [(0, 2, 3), (1, 5, 4), (0, 1, 4, 2), (0, 3, 5, 1), (2, 4, 5, 3)], [M['purpleLight'], M['purpleSide'], M['purpleEdge']], [0, 0, 1, 1, 1])
    bevel = obj.modifiers.new('Soft machined edge', 'BEVEL')
    bevel.width = .017
    bevel.segments = 3
    bevel.material = 2
    obj.modifiers.new('Weighted arrowhead normals', 'WEIGHTED_NORMAL')


def focus_mark(center=(.08, -.91, -.015)):
    # A purple glossy surface seed matches the approved reference, without glow.
    c = Vector(center)
    glossy_ball('Focus | dark purple surround', c, .166, M['purpleDark'])
    glossy_ball('Focus | polished lavender seed', c + Vector((0, -.063, 0)), .142, M['purpleLight'])


def icon_focus():
    s = scene('01 | MFO Focus Surface')
    sculpt_form('Reference | faceted sculpt surface')
    orbit()
    focus_mark()
    return s, 'mfo-focus-surface.png'


def icon_face_set():
    s = scene('02 | Face Set MFO')
    sculpt_form('Reference | contiguous purple Face Set', region=True)
    orbit(face_set=True)
    return s, 'mfo-face-set.png'


def icon_fill():
    s = scene('03 | Smart Face Set Fill', scale=4.15, camera=(-3.7, -5.3, 8), target=(0, 0, .02))
    inner = [(-.82, -.78), (0, -.90), (.88, -.74), (.92, 0), (.76, .89), (0, .94), (-.88, .87), (-.95, 0)]
    outer = [(-1.38, -1.25), (0, -1.33), (1.39, -1.22), (1.44, 0), (1.30, 1.29), (0, 1.43), (-1.43, 1.22), (-1.48, 0)]
    def surface_z(x, y):
        return .14 - .09*x*x - .025*y*y + .045*x*y
    verts = [(0, 0, .23)] + [(x, y, surface_z(x, y)) for x, y in inner + outer]
    faces, indices = [], []
    for j in range(8):
        faces.append((0, 1+j, 1+(j+1)%8))
        indices.append(1 if j in (4, 5, 6) else (2 if j in (2, 3) else 3))
    for j in range(8):
        a, b = 1+j, 1+(j+1)%8
        faces.extend(((a, 9+j, 9+(j+1)%8), (a, 9+(j+1)%8, b)))
        indices.extend((0, 0))
    obj = mesh('Fill | bounded face-set surface', verts, faces, [M['clay'], M['green'], M['yellow'], M['red'], M['clayDark']], indices)
    solid = obj.modifiers.new('Surface edge thickness', 'SOLIDIFY')
    solid.thickness = .075
    solid.material_offset_rim = 4
    mesh_edges(obj, .010)
    glossy_ball('Fill | seed dark rim', (0, -.04, .29), .140, M['clayDark'])
    glossy_ball('Fill | white seed', (0, -.075, .335), .116, M['white'])
    return s, 'mfo-smart-face-set-fill.png'


def ridge_pos(x, offset):
    return Vector((x, .29 * math.sin(x * 1.6) + offset, -.065 * x * x))


def icon_ridge():
    s = scene('04 | Guided Ridge', scale=4.35, camera=(6.0, -6.8, 5.3), target=(0, 0, .28))
    xs = [-1.52, -.98, -.40, .16, .77, 1.42]
    profile = [.66, 1.10, 1.19, .95, .51, .17]
    offsets = [-1.02, -.51, 0, .52, 1.02]
    heights = [0, .29, .98, .30, 0]
    verts, faces, idx = [], [], []
    for xi, x in enumerate(xs):
        for offset, height in zip(offsets, heights):
            p = ridge_pos(x, offset)
            p.z += height * profile[xi]
            verts.append(p)
    w = len(offsets)
    for i in range(len(xs) - 1):
        for j in range(w - 1):
            a = i * w + j
            b = (i + 1) * w + j
            if (i+j)%2:
                faces.extend(((a, b, a+1), (b, b+1, a+1)))
            else:
                faces.extend(((a, b, b+1), (a, b+1, a+1)))
            idx.extend((0, 0))
    perimeter = list(range(w)) + [i*w+w-1 for i in range(1, len(xs))] + [(len(xs)-1)*w+j for j in range(w-2, -1, -1)] + [i*w for i in range(len(xs)-2, 0, -1)]
    lower = []
    for index in perimeter:
        p = Vector(verts[index])
        p.z = -.45
        lower.append(len(verts))
        verts.append(p)
    for i, top in enumerate(perimeter):
        faces.append((top, lower[i], lower[(i+1)%len(lower)], perimeter[(i+1)%len(perimeter)]))
        idx.append(2)
    faces.append(tuple(reversed(lower)))
    idx.append(2)
    obj = mesh('Ridge | continuous bent quad surface', verts, faces, [M['clay'], M['clayLight'], M['clayDark']], idx)
    mesh_edges(obj, .009)
    points = []
    for xi, x in enumerate(xs):
        p = ridge_pos(x, 0)
        p.z += .98 * profile[xi] + .045
        points.append(p)
    ribbon('Guide | lavender rail directly on the crest', points, [(0, 1, 0)]*len(points), .235, .065)
    return s, 'mfo-guided-ridge.png'


def tube_center(t):
    p0 = Vector((-1.50, 0, -.35))
    p1 = Vector((-.10, 0, -.55))
    p2 = Vector((1.30, 0, -.25))
    p3 = Vector((1.14, 0, .97))
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3


def tube_frame(t):
    tangent = (tube_center(min(1, t + .001)) - tube_center(max(0, t - .001))).normalized()
    a = Vector((tangent.z, 0, -tangent.x)).normalized()
    b = Vector((0, 1, 0))
    return a, b


def tube_radius(t):
    return .60 - .035 * t


def tube_ring(t, extra=0):
    c = tube_center(t)
    a, b = tube_frame(t)
    r = tube_radius(t) + extra
    return [c + r * (math.cos(TAU * j / 12) * a + .86 * math.sin(TAU * j / 12) * b) for j in range(12)]


def icon_tube():
    s = scene('05 | Tube Shape', scale=3.65, camera=(-7.4, -8, 3.8), target=(-.10, 0, .20))
    rings = [tube_ring(i / 9) for i in range(10)]
    verts = [p for ring in rings for p in ring]
    faces, ids = [], []
    for i in range(9):
        for j in range(12):
            faces.append((i*12+j, i*12+(j+1)%12, (i+1)*12+(j+1)%12, (i+1)*12+j))
            ids.append(0)
    start_center = len(verts)
    verts.append(tube_center(0))
    end_center = len(verts)
    verts.append(tube_center(1))
    for j in range(12):
        faces.extend(((start_center, (j+1)%12, j), (end_center, 9*12+j, 9*12+(j+1)%12)))
        ids.extend((0, 0))
    obj = mesh('Tube | bent sculpt bundle with visible end profile', verts, faces, [M['clay'], M['clayLight'], M['clayDark']], ids)
    mesh_edges(obj, .0085)
    # Actual cross sections follow local tangent, so the shaping cue stays honest.
    for n, t in enumerate((.155, .79)):
        c = tube_center(t)
        a, b = tube_frame(t)
        tangent = (tube_center(t+.001)-tube_center(t-.001)).normalized()
        centers = []
        for k in range(97):
            angle = TAU*k/96
            centers.append(c + (tube_radius(t)+.025)*(math.cos(angle)*a + .86*math.sin(angle)*b))
        ribbon('Shape | local cross-section band %d' % n, centers, [tangent]*len(centers), .17, .055)
    return s, 'mfo-tube-shape.png'


BUILDERS = {'focus': icon_focus, 'face_set': icon_face_set, 'fill': icon_fill, 'ridge': icon_ridge, 'tube': icon_tube}
selected = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else list(BUILDERS)
default_scene = bpy.context.scene
for name in selected:
    s, filename = BUILDERS[name]()
    s.render.filepath = os.path.join(OUT, filename)
    bpy.ops.render.render(write_still=True, scene=s.name)
    print('ASTRA_RENDER', s.render.filepath, flush=True)
if default_scene.name in bpy.data.scenes and len(bpy.data.scenes) > 1:
    bpy.data.scenes.remove(default_scene)
if len(selected) == 5:
    bpy.context.window.scene = bpy.data.scenes['01 | MFO Focus Surface']
    bpy.data.orphans_purge(do_recursive=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, 'mfo-toolbar-icons-astra.blend'))
print('ASTRA_COMPLETE', OUT, flush=True)
