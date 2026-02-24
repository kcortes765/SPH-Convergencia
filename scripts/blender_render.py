"""
blender_render.py — Render fotorrealista de simulacion SPH en Blender Cycles.

Convierte VTKs a PLY, importa en Blender, aplica materiales, y renderiza.

Ejecutar:
  blender --background --python blender_render.py
  blender --python blender_render.py           (con GUI para ver resultado)

Autor: Kevin Cortes (UCN 2026)
"""

import bpy
import os
import glob
import math

# === CONFIGURACION ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RENDER_DIR = os.path.join(PROJECT_ROOT, "data", "render", "dp004")
PLY_DIR = os.path.join(RENDER_DIR, "ply")
OUTPUT_DIR = os.path.join(RENDER_DIR, "blender_output")

CANAL_STL = os.path.join(PROJECT_ROOT, "models", "Canal_Playa_1esa20_750cm.stl")
BOULDER_STL = os.path.join(PROJECT_ROOT, "models", "BLIR3.stl")

# Frame a renderizar (5 = ~t=2.5s, impacto)
HERO_FRAME = 5
RENDER_SAMPLES = 256  # Cycles samples (mas = mejor calidad, mas lento)
RESOLUTION = (1920, 1080)


def clear_scene():
    """Elimina todos los objetos de la escena."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    # Limpiar datos huerfanos
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def setup_renderer():
    """Configura Cycles con GPU."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.cycles.samples = RENDER_SAMPLES
    scene.cycles.use_denoising = True

    # Intentar usar GPU
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = True
    scene.cycles.device = 'GPU'
    print(f"  Renderer: Cycles GPU, {RENDER_SAMPLES} samples, {RESOLUTION[0]}x{RESOLUTION[1]}")


def create_water_material():
    """Material de agua semi-transparente con refraccion."""
    mat = bpy.data.materials.new(name="Water")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Limpiar nodos default
    for node in nodes:
        nodes.remove(node)

    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    # Mix de Glass + Transparent para agua
    mix = nodes.new('ShaderNodeMixShader')
    mix.location = (200, 0)
    mix.inputs[0].default_value = 0.3  # 30% transparente

    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.location = (0, 100)
    glass.inputs['Color'].default_value = (0.4, 0.6, 0.8, 1.0)  # Azul agua
    glass.inputs['Roughness'].default_value = 0.05
    glass.inputs['IOR'].default_value = 1.33  # IOR del agua

    transparent = nodes.new('ShaderNodeBsdfTransparent')
    transparent.location = (0, -100)
    transparent.inputs['Color'].default_value = (0.6, 0.8, 1.0, 1.0)

    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(glass.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs[0])

    return mat


def create_rock_material():
    """Material de roca/boulder."""
    mat = bpy.data.materials.new(name="Rock")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (400, 0)

    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    principled.inputs['Base Color'].default_value = (0.35, 0.30, 0.25, 1.0)  # Gris-marron
    principled.inputs['Roughness'].default_value = 0.85
    principled.inputs['Specular IOR Level'].default_value = 0.3

    # Textura de ruido para variacion
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-200, 0)
    noise.inputs['Scale'].default_value = 50.0
    noise.inputs['Detail'].default_value = 8.0

    color_ramp = nodes.new('ShaderNodeValToRGB')
    color_ramp.location = (-50, 200)
    color_ramp.color_ramp.elements[0].color = (0.25, 0.20, 0.18, 1.0)
    color_ramp.color_ramp.elements[1].color = (0.45, 0.38, 0.30, 1.0)

    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs[0], output.inputs[0])

    return mat


def create_channel_material():
    """Material del canal (concreto/acero)."""
    mat = bpy.data.materials.new(name="Channel")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    principled = nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)  # Gris
    principled.inputs['Roughness'].default_value = 0.7
    principled.inputs['Metallic'].default_value = 0.1

    return mat


def setup_lighting():
    """Configura iluminacion tipo estudio."""
    # Luz principal (sol)
    bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    # Luz de relleno
    bpy.ops.object.light_add(type='AREA', location=(12, 3, 5))
    fill = bpy.context.object
    fill.data.energy = 200
    fill.data.size = 5

    # HDRI de fondo (cielo simple)
    world = bpy.data.worlds['World']
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.7, 0.8, 1.0, 1.0)  # Cielo azul claro
    bg.inputs['Strength'].default_value = 0.5

    print("  Iluminacion configurada (sol + fill + cielo)")


def setup_camera():
    """Configura camara con vista lateral del impacto."""
    bpy.ops.object.camera_add(location=(8.5, -4.0, 1.2))
    cam = bpy.context.object
    cam.rotation_euler = (math.radians(75), 0, math.radians(0))

    # Apuntar al boulder
    constraint = cam.constraints.new(type='TRACK_TO')
    empty = bpy.ops.object.empty_add(location=(8.5, 0.5, 0.15))
    target = bpy.context.object
    target.name = "CameraTarget"
    constraint.target = target
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    bpy.context.scene.camera = cam
    print("  Camara configurada (vista lateral del impacto)")


def import_water_surface(frame_idx):
    """Importa la isosurface del agua para un frame especifico."""
    ply_file = os.path.join(PLY_DIR, f"water_{frame_idx:04d}.ply")

    if os.path.exists(ply_file):
        bpy.ops.wm.ply_import(filepath=ply_file)
        obj = bpy.context.object
        obj.name = f"Water_f{frame_idx}"
        mat = create_water_material()
        obj.data.materials.append(mat)
        # Suavizar
        bpy.ops.object.shade_smooth()
        print(f"  Agua frame {frame_idx} importada desde PLY")
        return obj

    # Fallback: intentar VTK directo (requiere addon)
    vtk_files = sorted(glob.glob(os.path.join(RENDER_DIR, "surface", "WaterSurface_*.vtk")))
    if frame_idx < len(vtk_files):
        print(f"  AVISO: PLY no encontrado. Necesita conversion previa.")
        print(f"  Ejecutar primero: python convert_vtk_to_ply.py")
    return None


def import_boulder():
    """Importa el boulder desde STL."""
    if os.path.exists(BOULDER_STL):
        bpy.ops.wm.stl_import(filepath=BOULDER_STL)
        obj = bpy.context.object
        obj.name = "Boulder"
        # Escalar y posicionar (mismos valores que geometry_builder)
        obj.scale = (0.04, 0.04, 0.04)
        obj.location = (8.5, 0.5, 0.1)
        mat = create_rock_material()
        obj.data.materials.append(mat)
        bpy.ops.object.shade_smooth()
        print(f"  Boulder importado y posicionado")
        return obj
    else:
        print(f"  AVISO: Boulder STL no encontrado en {BOULDER_STL}")
        return None


def import_channel():
    """Importa el canal desde STL."""
    if os.path.exists(CANAL_STL):
        bpy.ops.wm.stl_import(filepath=CANAL_STL)
        obj = bpy.context.object
        obj.name = "Channel"
        mat = create_channel_material()
        obj.data.materials.append(mat)
        print(f"  Canal importado")
        return obj
    else:
        print(f"  AVISO: Canal STL no encontrado en {CANAL_STL}")
        return None


def add_floor():
    """Agrega un plano de fondo."""
    bpy.ops.mesh.primitive_plane_add(size=30, location=(7.5, 0.5, -0.2))
    floor = bpy.context.object
    floor.name = "Floor"
    mat = bpy.data.materials.new(name="Floor")
    mat.use_nodes = True
    principled = mat.node_tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (0.15, 0.15, 0.18, 1.0)
    principled.inputs['Roughness'].default_value = 0.9
    floor.data.materials.append(mat)


def render_frame(frame_idx, output_path=None):
    """Renderiza un frame y guarda como PNG."""
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"render_{frame_idx:04d}.png")

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"\n  Render guardado: {output_path}")


# === MAIN ===
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BLENDER RENDER — SPH-IncipientMotion")
    print("=" * 60)

    print("\n[1/6] Limpiando escena...")
    clear_scene()

    print("\n[2/6] Configurando renderer...")
    setup_renderer()

    print("\n[3/6] Importando geometria...")
    import_channel()
    import_boulder()
    water = import_water_surface(HERO_FRAME)

    if water is None:
        print("\n" + "!" * 60)
        print("  NECESITAS CONVERTIR VTK A PLY PRIMERO")
        print("  Ejecuta en PowerShell:")
        print("    python convert_vtk_to_ply.py")
        print("  Luego vuelve a correr este script.")
        print("!" * 60)

    print("\n[4/6] Configurando iluminacion...")
    setup_lighting()

    print("\n[5/6] Configurando camara...")
    setup_camera()
    add_floor()

    print("\n[6/6] Renderizando hero shot (frame 5, ~t=2.5s)...")
    if water:
        render_frame(HERO_FRAME)

    print("\n" + "=" * 60)
    print("  COMPLETADO")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
