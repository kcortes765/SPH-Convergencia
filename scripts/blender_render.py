"""
blender_render.py — Render fotorrealista de simulacion SPH en Blender Cycles.

Importa isosurfaces PLY del agua, STLs de canal/boulder, aplica materiales,
anima el boulder con datos Chrono, y renderiza.

IMPORTANTE: Usa CPU por defecto (las 50 sims LHS ocupan la GPU).
            Cambiar USE_GPU = True solo si la GPU esta libre.

Ejecutar en Blender:
  Opcion A (GUI):  Scripting tab > Open > Run Script
  Opcion B (headless):
    blender --background --python scripts/blender_render.py
    blender --background --python scripts/blender_render.py -- --all-frames
    blender --background --python scripts/blender_render.py -- --gpu

Autor: Kevin Cortes (UCN 2026)
Fecha: 2026-02-24
"""

import bpy
import os
import sys
import csv
import glob
import math

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACION
# ═══════════════════════════════════════════════════════════════════════

# PROJECT_ROOT = directorio raiz del proyecto (padre de scripts/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-detectar directorio de render (el mas reciente en data/render/)
_RENDER_BASE = os.path.join(PROJECT_ROOT, "data", "render")
def _find_render_dir():
    if not os.path.exists(_RENDER_BASE):
        return os.path.join(_RENDER_BASE, "dp004")  # fallback
    subdirs = [d for d in os.listdir(_RENDER_BASE)
               if os.path.isdir(os.path.join(_RENDER_BASE, d))]
    if not subdirs:
        return os.path.join(_RENDER_BASE, "dp004")
    subdirs.sort(key=lambda d: os.path.getmtime(os.path.join(_RENDER_BASE, d)), reverse=True)
    return os.path.join(_RENDER_BASE, subdirs[0])

RENDER_DIR = _find_render_dir()
PLY_DIR = os.path.join(RENDER_DIR, "ply")
OUTPUT_DIR = os.path.join(RENDER_DIR, "blender_output")
CHRONO_CSV = os.path.join(RENDER_DIR, "csv", "ChronoExchange_mkbound_51.csv")

CANAL_STL = os.path.join(PROJECT_ROOT, "models", "Canal_Playa_1esa20_750cm.stl")
BOULDER_STL = os.path.join(PROJECT_ROOT, "models", "BLIR3.stl")

# Render settings
HERO_FRAME = 5              # Frame del impacto (~t=2.5s con TimeOut=0.5)
RENDER_SAMPLES = 256        # Cycles samples (256 + denoiser = buena calidad)
RESOLUTION = (1920, 1080)   # 1080p
USE_GPU = False             # False = CPU (seguro mientras sims corren)

# Boulder initial position/scale (mismos valores que geometry_builder)
BOULDER_SCALE = 0.04
BOULDER_INIT_POS = (8.487, 0.523, 0.124)
BOULDER_INIT_ROT = (0.0, 0.0, 0.0)

# Simulation timing
TIME_OUT = 0.5              # Intervalo entre snapshots (s)
FT_PAUSE = 0.5              # Tiempo de asentamiento (s)


# ═══════════════════════════════════════════════════════════════════════
# PARSE ARGS (despues de --)
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    """Parsea argumentos pasados despues de -- en la linea de comando de Blender."""
    argv = sys.argv
    if "--" in argv:
        args = argv[argv.index("--") + 1:]
    else:
        args = []
    return {
        'all_frames': '--all-frames' in args,
        'gpu': '--gpu' in args,
    }


# ═══════════════════════════════════════════════════════════════════════
# ESCENA
# ═══════════════════════════════════════════════════════════════════════

def clear_scene():
    """Elimina todos los objetos de la escena."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for mat in bpy.data.materials:
        if mat.users == 0:
            bpy.data.materials.remove(mat)


def setup_renderer(use_gpu=False):
    """Configura Cycles. CPU por defecto, GPU opcional."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True

    # Light paths para agua transparente
    scene.cycles.max_bounces = 12
    scene.cycles.transparent_max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.glossy_bounces = 8

    # Film transparente (para compositing posterior si se quiere)
    scene.render.film_transparent = False

    # Color management
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium Contrast'

    if use_gpu:
        try:
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            prefs.get_devices()
            for device in prefs.devices:
                device.use = True
            scene.cycles.device = 'GPU'
            print(f"  Renderer: Cycles GPU, {RENDER_SAMPLES} samples")
        except Exception as e:
            print(f"  GPU no disponible ({e}), usando CPU")
            scene.cycles.device = 'CPU'
    else:
        scene.cycles.device = 'CPU'
        print(f"  Renderer: Cycles CPU, {RENDER_SAMPLES} samples")

    print(f"  Resolucion: {RESOLUTION[0]}x{RESOLUTION[1]}")
    print(f"  Transmission bounces: 12 (para agua transparente)")


# ═══════════════════════════════════════════════════════════════════════
# MATERIALES
# ═══════════════════════════════════════════════════════════════════════

def create_water_material():
    """Material de agua con Principled BSDF + absorcion volumetrica."""
    mat = bpy.data.materials.new(name="Water")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    # Surface: Principled BSDF configurado como agua
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (200, 100)
    principled.inputs['Base Color'].default_value = (0.05, 0.15, 0.25, 1.0)
    principled.inputs['Roughness'].default_value = 0.02
    principled.inputs['IOR'].default_value = 1.333
    principled.inputs['Transmission Weight'].default_value = 1.0

    # Volume: absorcion para color azul-verdoso en profundidad
    vol_absorb = nodes.new('ShaderNodeVolumeAbsorption')
    vol_absorb.location = (200, -150)
    vol_absorb.inputs['Color'].default_value = (0.4, 0.75, 0.9, 1.0)
    vol_absorb.inputs['Density'].default_value = 2.0

    # Bump sutil para micro-ondas en la superficie
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-200, -50)
    noise.inputs['Scale'].default_value = 500.0
    noise.inputs['Detail'].default_value = 6.0

    bump = nodes.new('ShaderNodeBump')
    bump.location = (0, -50)
    bump.inputs['Strength'].default_value = 0.015

    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], principled.inputs['Normal'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    links.new(vol_absorb.outputs['Volume'], output.inputs['Volume'])

    return mat


def create_rock_material():
    """Material de roca caliza con textura procedural."""
    mat = bpy.data.materials.new(name="Rock")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    principled.inputs['Roughness'].default_value = 0.85
    principled.inputs['Specular IOR Level'].default_value = 0.3

    # Color base con variacion
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-200, 100)
    noise.inputs['Scale'].default_value = 50.0
    noise.inputs['Detail'].default_value = 8.0

    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.location = (0, 100)
    color_ramp.color_ramp.elements[0].color = (0.25, 0.20, 0.18, 1.0)
    color_ramp.color_ramp.elements[1].color = (0.45, 0.38, 0.30, 1.0)

    # Bump con Voronoi para rugosidad de roca
    voronoi = nodes.new('ShaderNodeTexVoronoi')
    voronoi.location = (-200, -100)
    voronoi.inputs['Scale'].default_value = 30.0

    rock_bump = nodes.new('ShaderNodeBump')
    rock_bump.location = (100, -100)
    rock_bump.inputs['Strength'].default_value = 0.3

    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(voronoi.outputs['Distance'], rock_bump.inputs['Height'])
    links.new(rock_bump.outputs['Normal'], principled.inputs['Normal'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    return mat


def create_channel_material():
    """Material del canal (concreto gris)."""
    mat = bpy.data.materials.new(name="Channel")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    principled = nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (0.50, 0.50, 0.48, 1.0)
    principled.inputs['Roughness'].default_value = 0.88
    principled.inputs['Metallic'].default_value = 0.0

    # Bump sutil
    musgrave = nodes.new('ShaderNodeTexMusgrave')
    musgrave.location = (-200, -100)
    musgrave.inputs['Scale'].default_value = 20.0

    ch_bump = nodes.new('ShaderNodeBump')
    ch_bump.location = (0, -100)
    ch_bump.inputs['Strength'].default_value = 0.08

    links.new(musgrave.outputs['Fac'], ch_bump.inputs['Height'])
    links.new(ch_bump.outputs['Normal'], principled.inputs['Normal'])

    return mat


# ═══════════════════════════════════════════════════════════════════════
# ILUMINACION Y CAMARA
# ═══════════════════════════════════════════════════════════════════════

def setup_lighting():
    """Iluminacion: sol + fill + HDRI cielo."""
    # Sol principal
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.name = "Sun_Key"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(1.0)  # Sombras suaves
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    # Fill lateral
    bpy.ops.object.light_add(type='AREA', location=(12, 3, 4))
    fill = bpy.context.object
    fill.name = "Fill_Light"
    fill.data.energy = 200
    fill.data.size = 5
    fill.data.color = (0.85, 0.90, 1.0)  # Ligeramente azul

    # Rim light (contraluz para resaltar splash)
    bpy.ops.object.light_add(type='SPOT', location=(10, 0.5, 3))
    rim = bpy.context.object
    rim.name = "Rim_Light"
    rim.data.energy = 400
    rim.data.spot_size = math.radians(45)
    rim.rotation_euler = (math.radians(60), 0, math.radians(-10))

    # HDRI cielo
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.65, 0.78, 1.0, 1.0)
    bg.inputs['Strength'].default_value = 0.5

    print("  Iluminacion: sol + fill azul + rim + cielo HDRI")


def setup_camera():
    """Camara 3/4 apuntando al boulder."""
    bpy.ops.object.camera_add(location=(6.5, -3.5, 1.5))
    cam = bpy.context.object
    cam.name = "Camera_Hero"
    cam.data.lens = 35  # 35mm

    # Target en el boulder
    bpy.ops.object.empty_add(location=(8.5, 0.5, 0.15))
    target = bpy.context.object
    target.name = "CameraTarget"

    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    bpy.context.scene.camera = cam
    print(f"  Camara: 35mm en ({6.5}, {-3.5}, {1.5}) -> boulder")


# ═══════════════════════════════════════════════════════════════════════
# IMPORTAR GEOMETRIA
# ═══════════════════════════════════════════════════════════════════════

def import_water_surface(frame_idx):
    """Importa la isosurface del agua (PLY) para un frame."""
    ply_file = os.path.join(PLY_DIR, f"water_{frame_idx:04d}.ply")

    if not os.path.exists(ply_file):
        print(f"  AVISO: PLY no encontrado: {ply_file}")
        print(f"  Ejecutar primero: python scripts/convert_vtk_to_ply.py")
        return None

    bpy.ops.wm.ply_import(filepath=ply_file)
    obj = bpy.context.object
    obj.name = f"Water_f{frame_idx}"
    mat = create_water_material()
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()

    n_verts = len(obj.data.vertices)
    print(f"  Agua frame {frame_idx}: {n_verts:,} vertices importados")
    return obj


def import_boulder():
    """Importa el boulder STL, escala y posiciona."""
    if not os.path.exists(BOULDER_STL):
        print(f"  AVISO: Boulder no encontrado: {BOULDER_STL}")
        return None

    bpy.ops.wm.stl_import(filepath=BOULDER_STL)
    obj = bpy.context.object
    obj.name = "Boulder"
    obj.scale = (BOULDER_SCALE, BOULDER_SCALE, BOULDER_SCALE)
    obj.location = BOULDER_INIT_POS
    mat = create_rock_material()
    obj.data.materials.append(mat)
    bpy.ops.object.shade_smooth()

    print(f"  Boulder: scale={BOULDER_SCALE}, pos={BOULDER_INIT_POS}")
    return obj


def import_channel():
    """Importa el canal STL."""
    if not os.path.exists(CANAL_STL):
        print(f"  AVISO: Canal no encontrado: {CANAL_STL}")
        return None

    bpy.ops.wm.stl_import(filepath=CANAL_STL)
    obj = bpy.context.object
    obj.name = "Channel"
    mat = create_channel_material()
    obj.data.materials.append(mat)

    n_faces = len(obj.data.polygons)
    print(f"  Canal: {n_faces:,} caras importadas")
    return obj


def add_floor():
    """Plano de fondo debajo del canal."""
    bpy.ops.mesh.primitive_plane_add(size=30, location=(7.5, 0.5, -0.2))
    floor = bpy.context.object
    floor.name = "Floor"
    mat = bpy.data.materials.new(name="Floor")
    mat.use_nodes = True
    principled = mat.node_tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (0.12, 0.12, 0.15, 1.0)
    principled.inputs['Roughness'].default_value = 0.95
    floor.data.materials.append(mat)


# ═══════════════════════════════════════════════════════════════════════
# ANIMACION DEL BOULDER CON DATOS CHRONO
# ═══════════════════════════════════════════════════════════════════════

def animate_boulder_from_chrono(boulder_obj):
    """
    Lee ChronoExchange CSV y aplica keyframes de posicion al boulder.

    El CSV tiene columnas: time, fcenter.x, fcenter.y, fcenter.z, etc.
    Los snapshots de render se graban cada TIME_OUT=0.5s, asi que
    mapeamos time -> frame_index.
    """
    if not os.path.exists(CHRONO_CSV):
        print(f"  AVISO: Chrono CSV no encontrado: {CHRONO_CSV}")
        print(f"  El boulder quedara estatico en su posicion inicial.")
        return False

    # Leer CSV
    times = []
    positions = []
    with open(CHRONO_CSV, 'r') as f:
        reader = csv.DictReader(f, delimiter=';')
        headers = reader.fieldnames
        # Buscar columnas de posicion
        cx_col = next((h for h in headers if 'fcenter.x' in h.lower()), None)
        cy_col = next((h for h in headers if 'fcenter.y' in h.lower()), None)
        cz_col = next((h for h in headers if 'fcenter.z' in h.lower()), None)
        time_col = next((h for h in headers if 'time' in h.lower()), None)

        if not all([cx_col, cy_col, cz_col, time_col]):
            print(f"  AVISO: Columnas Chrono no encontradas. Headers: {headers[:5]}")
            return False

        for row in reader:
            t = float(row[time_col])
            times.append(t)
            positions.append((
                float(row[cx_col]),
                float(row[cy_col]),
                float(row[cz_col]),
            ))

    if not times:
        print(f"  AVISO: Chrono CSV vacio")
        return False

    # Mapear: cada snapshot de render corresponde a t = frame_idx * TIME_OUT
    # Encontrar la posicion mas cercana en el CSV para cada frame
    ply_files = sorted(glob.glob(os.path.join(PLY_DIR, "water_*.ply")))
    n_frames = len(ply_files) if ply_files else 21

    print(f"  Animando boulder con {len(times)} timesteps Chrono -> {n_frames} frames")

    for frame_idx in range(n_frames):
        t_target = frame_idx * TIME_OUT
        # Buscar timestep mas cercano
        best_i = min(range(len(times)), key=lambda i: abs(times[i] - t_target))
        pos = positions[best_i]

        boulder_obj.location = pos
        boulder_obj.keyframe_insert(data_path="location", frame=frame_idx)

    print(f"  Boulder animado: {n_frames} keyframes ({times[0]:.2f}s a {times[-1]:.2f}s)")
    return True


# ═══════════════════════════════════════════════════════════════════════
# RENDER
# ═══════════════════════════════════════════════════════════════════════

def render_frame(frame_idx, output_path=None):
    """Renderiza un frame y guarda como PNG."""
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"render_{frame_idx:04d}.png")

    bpy.context.scene.render.filepath = output_path
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_depth = '16'
    bpy.ops.render.render(write_still=True)
    print(f"  Render guardado: {output_path}")


def render_all_frames():
    """Renderiza todos los frames disponibles (animacion completa)."""
    ply_files = sorted(glob.glob(os.path.join(PLY_DIR, "water_*.ply")))
    if not ply_files:
        print("  ERROR: No hay PLYs. Ejecutar convert_vtk_to_ply.py primero.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    n_total = len(ply_files)
    print(f"\n  Renderizando {n_total} frames (animacion completa)...")
    print(f"  Estimado CPU: ~{n_total * 15}-{n_total * 20} min")

    for i in range(n_total):
        print(f"\n  --- Frame {i}/{n_total-1} (t={i * TIME_OUT:.1f}s) ---")

        # Borrar agua del frame anterior
        for obj in bpy.data.objects:
            if obj.name.startswith("Water_f"):
                bpy.data.objects.remove(obj, do_unlink=True)

        # Importar agua de este frame
        water = import_water_surface(i)
        if water is None:
            print(f"  Saltando frame {i} (sin PLY)")
            continue

        # Mover boulder al frame correcto
        bpy.context.scene.frame_set(i)

        render_frame(i)

    print(f"\n  Animacion completada: {n_total} frames en {OUTPUT_DIR}")
    print(f"  Para compilar video:")
    print(f"    ffmpeg -framerate 2 -i render_%04d.png -c:v libx264 -crf 18 -pix_fmt yuv420p render.mp4")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    use_gpu = args['gpu'] or USE_GPU
    all_frames = args['all_frames']

    print("\n" + "=" * 60)
    print("  BLENDER RENDER — SPH-IncipientMotion")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Render dir:   {RENDER_DIR}")
    print(f"  Device:       {'GPU' if use_gpu else 'CPU'}")
    print(f"  Mode:         {'Animacion completa' if all_frames else f'Hero shot (frame {HERO_FRAME})'}")

    print("\n[1/7] Limpiando escena...")
    clear_scene()

    print("\n[2/7] Configurando renderer...")
    setup_renderer(use_gpu)

    print("\n[3/7] Importando canal...")
    import_channel()

    print("\n[4/7] Importando boulder...")
    boulder = import_boulder()

    print("\n[5/7] Configurando iluminacion...")
    setup_lighting()
    add_floor()

    print("\n[6/7] Configurando camara...")
    setup_camera()

    # Animar boulder si hay datos Chrono
    if boulder:
        print("\n[6.5/7] Animando boulder con datos Chrono...")
        animate_boulder_from_chrono(boulder)

    if all_frames:
        print("\n[7/7] Renderizando animacion completa...")
        render_all_frames()
    else:
        print(f"\n[7/7] Importando agua frame {HERO_FRAME}...")
        water = import_water_surface(HERO_FRAME)

        if water is None:
            print("\n" + "!" * 60)
            print("  NECESITAS CONVERTIR VTK A PLY PRIMERO")
            print("  En PowerShell:")
            print("    python scripts/convert_vtk_to_ply.py")
            print("  Luego vuelve a correr este script.")
            print("!" * 60)
        else:
            # Posicionar boulder en el frame correcto
            bpy.context.scene.frame_set(HERO_FRAME)
            print(f"  Renderizando hero shot (frame {HERO_FRAME}, ~t={HERO_FRAME * TIME_OUT:.1f}s)...")
            render_frame(HERO_FRAME)

    print("\n" + "=" * 60)
    print("  COMPLETADO")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


# Blender ejecuta como __main__ cuando se pasa con --python
main()
