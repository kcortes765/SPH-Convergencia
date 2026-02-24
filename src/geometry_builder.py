"""
geometry_builder.py — Modulo 1 del Pipeline SPH-IncipientMotion

Genera XMLs de caso DualSPHysics a partir de un template + STL + parametros.
Calcula propiedades fisicas (centro de masa, inercia) con trimesh y las inyecta
en el XML, corrigiendo las anomalias detectadas en la auditoria forense.

Correcciones respecto al setup de Diego:
  - Inercia: calculada por trimesh (geometria continua), no por GenCase (particulas)
  - Centro de masa: calculado por trimesh, no por GenCase
  - FtPause: inyectado automaticamente (asentamiento gravitatorio)

Autor: Kevin Cortes (UCN 2026)
"""

import shutil
import logging
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import trimesh
from lxml import etree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclass de parametros
# ---------------------------------------------------------------------------

@dataclass
class CaseParams:
    """Parametros que definen un caso de simulacion DualSPHysics."""

    case_name: str
    dp: float                                   # Distancia entre particulas (m)
    dam_height: float = 0.3                     # Altura columna de agua Dam Break (m)
    dam_length: float = 3.0                     # Largo columna de agua (m, eje X)
    boulder_mass: float = 1.06053               # Masa real del boulder (kg)
    boulder_scale: float = 0.04                 # Factor de escala uniforme del STL
    boulder_pos: tuple = (8.5, 0.5, 0.1)        # Posicion en el dominio (drawmove)
    boulder_rot: tuple = (0.0, 0.0, 0.0)        # Rotacion euler XYZ en grados
    material: str = "pvc"                        # Material en Floating_Materials.xml
    mkbound: int = 51                            # MK del boulder
    time_max: float = 10.0                       # Duracion de simulacion (s)
    time_out: float = 10.0                       # Intervalo de output de particulas (s)
    ft_pause: float = 0.5                        # Tiempo de asentamiento (s)
    chrono_savedata: float = 0.001               # Intervalo de guardado Chrono CSV (s)


# ---------------------------------------------------------------------------
# Funciones de geometria
# ---------------------------------------------------------------------------

def compute_boulder_properties(stl_path: Path, scale: float, rotation_deg: tuple,
                               mass_kg: float) -> dict:
    """
    Carga un STL, aplica escala y rotacion, y calcula propiedades fisicas.

    El orden de transformaciones replica lo que DualSPHysics hace internamente:
    1. drawscale (escala uniforme)
    2. drawrotate (rotacion euler XYZ en grados)
    La traslacion (drawmove) se aplica despues, no afecta las propiedades locales.

    Args:
        stl_path: Ruta al archivo STL del boulder.
        scale: Factor de escala uniforme (ej: 0.04).
        rotation_deg: Tupla (rx, ry, rz) en grados, rotacion euler XYZ.
        mass_kg: Masa real del boulder en kg.

    Returns:
        dict con claves:
          - volume_m3: Volumen en m^3
          - density_kgm3: Densidad implicita (kg/m^3)
          - center_of_mass: np.array [cx, cy, cz] en coords locales (post-escala+rotacion)
          - inertia_tensor: np.array 3x3 del tensor de inercia (kg*m^2)
          - bbox_min: np.array [xmin, ymin, zmin] en coords locales
          - bbox_max: np.array [xmax, ymax, zmax] en coords locales
          - bbox_size: np.array [dx, dy, dz] en metros
          - d_eq: Diametro equivalente esferico (m)
          - mesh: Objeto trimesh transformado (para debug)
    """
    mesh = trimesh.load(str(stl_path))

    if not mesh.is_watertight:
        logger.warning(f"STL {stl_path.name} NO es watertight. "
                       "Volumen e inercia pueden ser incorrectos.")
    if not mesh.is_volume:
        logger.warning(f"STL {stl_path.name} NO es manifold. "
                       "Verificar en MeshLab/Blender.")

    # 1. Escala uniforme
    mesh.apply_scale(scale)

    # 2. Rotacion euler XYZ (solo si hay rotacion no nula)
    rx, ry, rz = rotation_deg
    if abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rz) > 1e-6:
        rot = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        mesh.apply_transform(transform)

    # 3. Propiedades fisicas
    volume = mesh.volume
    density = mass_kg / volume
    center_of_mass = mesh.center_mass

    # Tensor de inercia: trimesh asume densidad=1.
    # Para masa real: I = I_trimesh * (masa / volumen) = I_trimesh * densidad
    inertia_density1 = mesh.moment_inertia  # 3x3, densidad=1
    inertia_tensor = inertia_density1 * density

    bbox_min = mesh.bounds[0]
    bbox_max = mesh.bounds[1]
    bbox_size = bbox_max - bbox_min
    d_eq = (6.0 * volume / np.pi) ** (1.0 / 3.0)

    logger.info(f"Boulder '{stl_path.name}' (escala={scale}, rot={rotation_deg}):")
    logger.info(f"  Volumen:     {volume:.8f} m^3 ({volume*1000:.3f} L)")
    logger.info(f"  Densidad:    {density:.1f} kg/m^3")
    logger.info(f"  Centro masa: [{center_of_mass[0]:.6f}, {center_of_mass[1]:.6f}, "
                f"{center_of_mass[2]:.6f}] m (local)")
    logger.info(f"  d_eq:        {d_eq:.4f} m ({d_eq*100:.1f} cm)")
    logger.info(f"  Bbox size:   {bbox_size[0]:.4f} x {bbox_size[1]:.4f} x "
                f"{bbox_size[2]:.4f} m")
    logger.info(f"  dim_min/dp:  (requiere dp para calcular)")

    return {
        'volume_m3': volume,
        'density_kgm3': density,
        'center_of_mass': center_of_mass,
        'inertia_tensor': inertia_tensor,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'bbox_size': bbox_size,
        'd_eq': d_eq,
        'mesh': mesh,
    }


def compute_fillbox(boulder_props: dict, boulder_pos: tuple,
                    margin: float = 0.5) -> dict:
    """
    Calcula los parametros del fillbox en coordenadas del dominio.

    El fillbox debe:
    - Tener un seed point DENTRO del boulder (usamos centro de masa)
    - Cubrir todo el bounding box del boulder con margen

    Args:
        boulder_props: Dict retornado por compute_boulder_properties.
        boulder_pos: Tupla (x, y, z) de la posicion drawmove en el dominio.
        margin: Margen extra alrededor del bounding box (metros).

    Returns:
        dict con claves: seed_x/y/z, point_x/y/z, size_x/y/z
    """
    pos = np.array(boulder_pos)
    cm_local = boulder_props['center_of_mass']
    bb_min_local = boulder_props['bbox_min']
    bb_max_local = boulder_props['bbox_max']

    # Centro de masa en coordenadas del dominio = local + drawmove
    cm_domain = cm_local + pos

    # Bounding box en coordenadas del dominio
    bb_min_domain = bb_min_local + pos
    bb_max_domain = bb_max_local + pos

    # Fillbox region: bounding box + margen
    fb_point = bb_min_domain - margin
    fb_size = (bb_max_domain - bb_min_domain) + 2 * margin

    return {
        'seed_x': float(cm_domain[0]),
        'seed_y': float(cm_domain[1]),
        'seed_z': float(cm_domain[2]),
        'point_x': float(fb_point[0]),
        'point_y': float(fb_point[1]),
        'point_z': float(fb_point[2]),
        'size_x': float(fb_size[0]),
        'size_y': float(fb_size[1]),
        'size_z': float(fb_size[2]),
    }


# ---------------------------------------------------------------------------
# Reubicacion automatica de gauges
# ---------------------------------------------------------------------------

def relocate_gauges(tree: etree._ElementTree, boulder_props: dict,
                    boulder_pos: tuple, dp: float) -> None:
    """
    Reubica los gauges (velocity y maxz) del XML segun la posicion del boulder.

    Estrategia:
    - Gauges UPSTREAM del boulder (x < boulder_xmin): se mantienen en su x original
      pero z se baja a max(dp, 0.001) para garantizar contacto con el agua.
    - Gauges CERCA del boulder: se redistribuyen automaticamente:
      * V_pre: 1*dp antes de la cara frontal (upstream)
      * V_mid: en el centroide del boulder a media altura
      * V_post: 1*dp despues de la cara trasera (downstream)
    - Gauges maxz: distlimit se ajusta a max(0.04, 2*dp).

    Args:
        tree: Arbol lxml del XML.
        boulder_props: Dict de compute_boulder_properties.
        boulder_pos: Tupla (x, y, z) de drawmove.
        dp: Espaciamiento de particulas.
    """
    root = tree.getroot()
    gauges = root.find('.//gauges')
    if gauges is None:
        logger.warning("No se encontro <gauges> en el XML — salteando reubicacion.")
        return

    pos = np.array(boulder_pos)
    bb_min = boulder_props['bbox_min'] + pos
    bb_max = boulder_props['bbox_max'] + pos
    cm_domain = boulder_props['center_of_mass'] + pos

    boulder_xmin = float(bb_min[0])
    boulder_xmax = float(bb_max[0])
    boulder_y = float(cm_domain[1])
    boulder_zmid = float((bb_min[2] + bb_max[2]) / 2)
    floor_z = max(dp, 0.001)

    logger.info(f"  Reubicando gauges: boulder x=[{boulder_xmin:.3f}, {boulder_xmax:.3f}], "
                f"y={boulder_y:.3f}, z_mid={boulder_zmid:.3f}")

    # --- Velocity gauges ---
    vel_gauges = list(gauges.iter('velocity'))
    near_boulder = []
    for vg in vel_gauges:
        pt = vg.find('point')
        if pt is None:
            continue
        gx = float(pt.get('x', 0))
        # Gauge "cerca" del boulder: dentro de +-2m del centro del boulder
        if abs(gx - cm_domain[0]) < 2.0:
            near_boulder.append(vg)
        else:
            # Gauges lejos: solo bajar z al piso para asegurar contacto con agua
            old_z = float(pt.get('z', 0))
            pt.set('z', _fmt(floor_z, 4))
            if abs(old_z - floor_z) > 0.01:
                logger.info(f"    {vg.get('name')}: z {old_z:.3f} -> {floor_z:.4f} (al piso)")

    # Redistribuir los gauges cercanos al boulder
    if len(near_boulder) >= 3:
        # Gauge 0: upstream de la cara frontal
        pt0 = near_boulder[0].find('point')
        pt0.set('x', _fmt(boulder_xmin - dp, 4))
        pt0.set('y', _fmt(boulder_y, 4))
        pt0.set('z', _fmt(floor_z, 4))
        near_boulder[0].set('comment', 'Auto: upstream del boulder')
        logger.info(f"    {near_boulder[0].get('name')}: -> upstream x={boulder_xmin - dp:.4f}")

        # Gauge 1: centroide a media altura
        pt1 = near_boulder[1].find('point')
        pt1.set('x', _fmt(float(cm_domain[0]), 4))
        pt1.set('y', _fmt(boulder_y, 4))
        pt1.set('z', _fmt(boulder_zmid, 4))
        near_boulder[1].set('comment', 'Auto: centroide del boulder')
        logger.info(f"    {near_boulder[1].get('name')}: -> centroide x={cm_domain[0]:.4f}")

        # Gauge 2: downstream de la cara trasera
        pt2 = near_boulder[2].find('point')
        pt2.set('x', _fmt(boulder_xmax + dp, 4))
        pt2.set('y', _fmt(boulder_y, 4))
        pt2.set('z', _fmt(floor_z, 4))
        near_boulder[2].set('comment', 'Auto: downstream del boulder')
        logger.info(f"    {near_boulder[2].get('name')}: -> downstream x={boulder_xmax + dp:.4f}")

        # Gauges adicionales cerca: distribuir en Y (lateral izq/der)
        for i, vg in enumerate(near_boulder[3:], start=3):
            pt = vg.find('point')
            y_offset = 0.15 * (1 if i % 2 == 0 else -1)
            pt.set('x', _fmt(float(cm_domain[0]), 4))
            pt.set('y', _fmt(boulder_y + y_offset, 4))
            pt.set('z', _fmt(floor_z, 4))
            vg.set('comment', f'Auto: lateral boulder y_offset={y_offset:+.2f}')
    elif near_boulder:
        # Menos de 3 gauges cerca: solo bajar z
        for vg in near_boulder:
            pt = vg.find('point')
            pt.set('z', _fmt(floor_z, 4))

    # --- MaxZ gauges ---
    maxz_gauges = list(gauges.iter('maxz'))
    distlimit_val = _fmt(max(0.04, 2 * dp), 4)

    for mz in maxz_gauges:
        pt0 = mz.find('point0')
        if pt0 is None:
            continue
        gx = float(pt0.get('x', 0))

        # Ajustar distlimit
        dl = mz.find('distlimit')
        if dl is not None:
            dl.set('value', distlimit_val)

        # Gauges cerca del boulder: reubicar a cara frontal
        if abs(gx - cm_domain[0]) < 2.0:
            pt0.set('y', _fmt(boulder_y, 4))
            # z ligeramente sobre el piso
            pt0.set('z', _fmt(floor_z, 4))
            logger.info(f"    {mz.get('name')}: y->{boulder_y:.3f}, z->{floor_z:.4f}, "
                        f"distlimit->{distlimit_val}")
        else:
            # Lejos: bajar z al piso
            old_z = float(pt0.get('z', 0))
            if old_z > floor_z + 0.05:
                pt0.set('z', _fmt(floor_z + 0.05, 4))

    logger.info(f"  Gauges reubicados: {len(vel_gauges)} velocity, {len(maxz_gauges)} maxz")


# ---------------------------------------------------------------------------
# Funciones de modificacion XML
# ---------------------------------------------------------------------------

def _set_param(parameters_elem, key: str, value):
    """Modifica el value de un <parameter key="..."> existente."""
    for param in parameters_elem.findall('parameter'):
        if param.get('key') == key:
            param.set('value', str(value))
            return
    raise KeyError(f"Parametro '{key}' no encontrado en el XML")


def _fmt(value: float, precision: int = 8) -> str:
    """Formatea un float eliminando ceros trailing innecesarios."""
    formatted = f"{value:.{precision}f}"
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.')
    return formatted


def modify_xml(tree: etree._ElementTree, params: CaseParams,
               boulder_props: dict, fillbox: dict) -> etree._ElementTree:
    """
    Modifica el arbol XML del template con los parametros del caso.

    Inyecta:
    - dp (resolucion)
    - Altura del Dam Break
    - Transformaciones del boulder (scale, move, rotate)
    - Fillbox (seed y region)
    - Massbody, centro de masa, tensor de inercia
    - Material del boulder
    - FtPause, TimeMax, TimeOut
    - Tiempos de gauges

    Args:
        tree: Arbol lxml parseado del template XML.
        params: CaseParams con la configuracion del caso.
        boulder_props: Dict de propiedades del boulder (de compute_boulder_properties).
        fillbox: Dict de parametros del fillbox (de compute_fillbox).

    Returns:
        Arbol lxml modificado.
    """
    root = tree.getroot()

    # --- dp ---
    definition = root.find('.//geometry/definition')
    definition.set('dp', _fmt(params.dp, 4))

    # --- Dam Break (altura de la columna de agua) ---
    drawbox = root.find('.//drawbox')
    size_elem = drawbox.find('size')
    size_elem.set('x', _fmt(params.dam_length))
    size_elem.set('z', _fmt(params.dam_height))

    # --- Boulder: drawfilestl transforms ---
    drawfilestl = None
    for elem in root.iter('drawfilestl'):
        if 'BLIR' in elem.get('file', ''):
            drawfilestl = elem
            break
    if drawfilestl is None:
        raise ValueError("No se encontro <drawfilestl> del boulder en el XML")

    # drawscale
    drawscale = drawfilestl.find('drawscale')
    if drawscale is None:
        drawscale = etree.SubElement(drawfilestl, 'drawscale')
    s = _fmt(params.boulder_scale)
    drawscale.set('x', s)
    drawscale.set('y', s)
    drawscale.set('z', s)

    # drawrotate (solo si hay rotacion no nula)
    rx, ry, rz = params.boulder_rot
    has_rotation = abs(rx) > 1e-6 or abs(ry) > 1e-6 or abs(rz) > 1e-6
    drawrotate = drawfilestl.find('drawrotate')
    if has_rotation:
        if drawrotate is None:
            # Insertar ANTES de drawmove (orden: scale, rotate, move)
            drawmove = drawfilestl.find('drawmove')
            idx = list(drawfilestl).index(drawmove) if drawmove is not None else len(drawfilestl)
            drawrotate = etree.Element('drawrotate')
            drawfilestl.insert(idx, drawrotate)
        drawrotate.set('angx', _fmt(rx))
        drawrotate.set('angy', _fmt(ry))
        drawrotate.set('angz', _fmt(rz))
    elif drawrotate is not None:
        drawfilestl.remove(drawrotate)

    # drawmove
    drawmove = drawfilestl.find('drawmove')
    if drawmove is None:
        drawmove = etree.SubElement(drawfilestl, 'drawmove')
    bx, by, bz = params.boulder_pos
    drawmove.set('x', _fmt(bx))
    drawmove.set('y', _fmt(by))
    drawmove.set('z', _fmt(bz))

    # --- Fillbox ---
    fillbox_elem = None
    for fb in root.iter('fillbox'):
        if fb.get('mkbound') == str(params.mkbound):
            fillbox_elem = fb
            break
    if fillbox_elem is None:
        raise ValueError(f"No se encontro <fillbox mkbound='{params.mkbound}'> en el XML")

    fillbox_elem.set('x', _fmt(fillbox['seed_x'], 4))
    fillbox_elem.set('y', _fmt(fillbox['seed_y'], 4))
    fillbox_elem.set('z', _fmt(fillbox['seed_z'], 4))
    fb_point = fillbox_elem.find('point')
    fb_point.set('x', _fmt(fillbox['point_x'], 4))
    fb_point.set('y', _fmt(fillbox['point_y'], 4))
    fb_point.set('z', _fmt(fillbox['point_z'], 4))
    fb_size = fillbox_elem.find('size')
    fb_size.set('x', _fmt(fillbox['size_x'], 4))
    fb_size.set('y', _fmt(fillbox['size_y'], 4))
    fb_size.set('z', _fmt(fillbox['size_z'], 4))

    # --- Floating body: massbody, center, inertia ---
    floating = root.find('.//floatings/floating')
    if floating is None:
        raise ValueError("No se encontro <floating> en el XML")

    # Ajustar mkbound range al boulder actual
    floating.set('mkbound', str(params.mkbound))
    floating.set('property', params.material)

    # massbody
    massbody = floating.find('massbody')
    if massbody is None:
        massbody = etree.SubElement(floating, 'massbody')
    massbody.set('value', _fmt(params.boulder_mass, 5))
    massbody.tail = '\n\t\t\t\t'  # Limpiar el comentario malformado de Diego

    # center (en coordenadas del dominio)
    cm_domain = boulder_props['center_of_mass'] + np.array(params.boulder_pos)
    center = floating.find('center')
    if center is None:
        center = etree.SubElement(floating, 'center')
    center.set('x', _fmt(cm_domain[0], 6))
    center.set('y', _fmt(cm_domain[1], 6))
    center.set('z', _fmt(cm_domain[2], 6))
    center.tail = '\n\t\t\t\t'

    # inertia (diagonal Ixx, Iyy, Izz — formato Chrono: <inertia x="" y="" z=""/>)
    inertia_elem = floating.find('inertia')
    if inertia_elem is not None:
        floating.remove(inertia_elem)
    inertia_elem = etree.SubElement(floating, 'inertia')
    inertia_elem.tail = '\n\t\t\t'
    I = boulder_props['inertia_tensor']
    inertia_elem.set('x', f'{I[0][0]:.8g}')
    inertia_elem.set('y', f'{I[1][1]:.8g}')
    inertia_elem.set('z', f'{I[2][2]:.8g}')

    # --- Execution parameters ---
    parameters = root.find('.//execution/parameters')
    _set_param(parameters, 'FtPause', _fmt(params.ft_pause))
    _set_param(parameters, 'TimeMax', _fmt(params.time_max))
    _set_param(parameters, 'TimeOut', _fmt(params.time_out))

    # --- Chrono savedata interval ---
    chrono_savedata = root.find('.//chrono/savedata')
    if chrono_savedata is not None:
        chrono_savedata.set('value', _fmt(params.chrono_savedata))

    # --- Gauge times (ajustar end al TimeMax) ---
    for gauge_default in root.iter('default'):
        computetime = gauge_default.find('computetime')
        if computetime is not None:
            computetime.set('end', _fmt(params.time_max))
        outputtime = gauge_default.find('outputtime')
        if outputtime is not None:
            outputtime.set('end', _fmt(params.time_max))

    # --- Reubicar gauges segun posicion del boulder ---
    relocate_gauges(tree, boulder_props, params.boulder_pos, params.dp)

    return tree


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def build_case(template_xml: Path, boulder_stl: Path, beach_stl: Path,
               materials_xml: Path, output_dir: Path,
               params: CaseParams) -> Path:
    """
    Genera un caso completo de DualSPHysics listo para GenCase.

    Crea una carpeta con:
    - {case_name}_Def.xml (XML de definicion modificado)
    - BLIR3.stl (copia del boulder)
    - Canal_Playa_*.stl (copia de la playa)
    - Floating_Materials.xml (copia)

    Args:
        template_xml: Ruta al template XML base.
        boulder_stl: Ruta al STL del boulder.
        beach_stl: Ruta al STL de la playa.
        materials_xml: Ruta al Floating_Materials.xml.
        output_dir: Directorio donde crear la carpeta del caso.
        params: CaseParams con la configuracion.

    Returns:
        Path al XML de definicion generado.
    """
    # Crear directorio del caso
    case_dir = output_dir / params.case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Calcular propiedades del boulder
    logger.info(f"Calculando propiedades del boulder para caso '{params.case_name}'...")
    boulder_props = compute_boulder_properties(
        stl_path=boulder_stl,
        scale=params.boulder_scale,
        rotation_deg=params.boulder_rot,
        mass_kg=params.boulder_mass,
    )

    # Verificar resolucion
    dim_min = min(boulder_props['bbox_size'])
    particles_dim_min = dim_min / params.dp
    logger.info(f"  dp={params.dp}m -> dim_min/dp = {particles_dim_min:.1f} "
                f"({dim_min:.4f}m / {params.dp}m)")
    if particles_dim_min < 5:
        logger.warning(f"  ATENCION: Solo {particles_dim_min:.1f} particulas en la "
                       f"dimension minima del boulder. Recomendado: >=10.")

    # 2. Calcular fillbox
    fillbox = compute_fillbox(boulder_props, params.boulder_pos)
    logger.info(f"  Fillbox seed: ({fillbox['seed_x']:.4f}, {fillbox['seed_y']:.4f}, "
                f"{fillbox['seed_z']:.4f})")

    # 3. Parsear y modificar XML
    parser = etree.XMLParser(remove_blank_text=False)
    tree = etree.parse(str(template_xml), parser)
    tree = modify_xml(tree, params, boulder_props, fillbox)

    # 4. Escribir XML
    xml_filename = f"{params.case_name}_Def.xml"
    xml_path = case_dir / xml_filename
    tree.write(str(xml_path), xml_declaration=True, encoding='UTF-8',
               pretty_print=True)
    logger.info(f"  XML generado: {xml_path}")

    # 5. Copiar archivos auxiliares
    shutil.copy2(str(boulder_stl), str(case_dir / boulder_stl.name))
    shutil.copy2(str(beach_stl), str(case_dir / beach_stl.name))
    shutil.copy2(str(materials_xml), str(case_dir / materials_xml.name))
    logger.info(f"  Archivos copiados a {case_dir}")

    # 6. Guardar resumen de propiedades (util para debug y ETL)
    summary_path = case_dir / "boulder_properties.txt"
    _write_properties_summary(summary_path, params, boulder_props, fillbox)

    return xml_path


def _write_properties_summary(path: Path, params: CaseParams,
                              props: dict, fillbox: dict):
    """Escribe un archivo de texto con las propiedades calculadas del boulder."""
    cm = props['center_of_mass']
    cm_domain = cm + np.array(params.boulder_pos)
    I = props['inertia_tensor']

    lines = [
        f"# Boulder Properties — {params.case_name}",
        f"# Generado por geometry_builder.py",
        f"",
        f"stl_scale:       {params.boulder_scale}",
        f"stl_rotation:    {params.boulder_rot}",
        f"domain_position: {params.boulder_pos}",
        f"mass_kg:         {params.boulder_mass}",
        f"material:        {params.material}",
        f"dp:              {params.dp}",
        f"",
        f"volume_m3:       {props['volume_m3']:.8f}",
        f"density_kgm3:    {props['density_kgm3']:.1f}",
        f"d_eq_m:          {props['d_eq']:.6f}",
        f"bbox_size:       {props['bbox_size'][0]:.4f} x {props['bbox_size'][1]:.4f} x {props['bbox_size'][2]:.4f}",
        f"dim_min_m:       {min(props['bbox_size']):.4f}",
        f"dim_min_over_dp: {min(props['bbox_size']) / params.dp:.1f}",
        f"",
        f"center_mass_local:  [{cm[0]:.6f}, {cm[1]:.6f}, {cm[2]:.6f}]",
        f"center_mass_domain: [{cm_domain[0]:.6f}, {cm_domain[1]:.6f}, {cm_domain[2]:.6f}]",
        f"",
        f"inertia_tensor (kg*m^2):",
        f"  [{I[0][0]:.8g}, {I[0][1]:.8g}, {I[0][2]:.8g}]",
        f"  [{I[1][0]:.8g}, {I[1][1]:.8g}, {I[1][2]:.8g}]",
        f"  [{I[2][0]:.8g}, {I[2][1]:.8g}, {I[2][2]:.8g}]",
        f"",
        f"fillbox_seed:    ({fillbox['seed_x']:.4f}, {fillbox['seed_y']:.4f}, {fillbox['seed_z']:.4f})",
        f"fillbox_point:   ({fillbox['point_x']:.4f}, {fillbox['point_y']:.4f}, {fillbox['point_z']:.4f})",
        f"fillbox_size:    ({fillbox['size_x']:.4f}, {fillbox['size_y']:.4f}, {fillbox['size_z']:.4f})",
    ]

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# CLI: permite correr directamente para generar un caso de prueba
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import json

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    project_root = Path(__file__).parent.parent

    # Cargar configuracion
    config_path = project_root / 'config' / 'dsph_config.json'
    with open(config_path) as f:
        config = json.load(f)

    # Rutas
    template_xml = project_root / config['paths']['template_xml']
    boulder_stl = project_root / config['paths']['boulder_stl']
    beach_stl = project_root / config['paths']['beach_stl']
    materials_xml = project_root / config['paths']['materials_xml']
    cases_dir = project_root / config['paths']['cases_dir']

    # Caso de prueba: mismos parametros de Diego pero con FtPause=0.5
    # y dp=0.02 (desarrollo) en vez de dp=0.05 (inaceptable)
    test_params = CaseParams(
        case_name="test_diego_reference",
        dp=0.02,
        dam_height=0.3,
        boulder_mass=1.06053,
        boulder_scale=0.04,
        boulder_pos=(8.5, 0.5, 0.1),
        boulder_rot=(0.0, 0.0, 0.0),
        material="pvc",
        time_max=10.0,
        time_out=10.0,
        ft_pause=0.5,
    )

    xml_path = build_case(
        template_xml=template_xml,
        boulder_stl=boulder_stl,
        beach_stl=beach_stl,
        materials_xml=materials_xml,
        output_dir=cases_dir,
        params=test_params,
    )

    print(f"\nCaso generado exitosamente: {xml_path}")
    print(f"Carpeta del caso: {xml_path.parent}")
