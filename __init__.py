# ComfyUI custom node: Bake High to Active Low (runs your Blender in background)

import json
import os
import subprocess
import sys
import tempfile
import uuid

import folder_paths
import numpy as np
import torch

try:
    from PIL import Image
except Exception:
    Image = None


def _resolve_path_string(raw):
    raw = str(raw)
    if os.path.isabs(raw):
        return raw

    try:
        return folder_paths.get_3d_path(raw)
    except Exception:
        return os.path.join(folder_paths.get_input_directory(), raw)


def _extract_3d_path(value, out_dir, tag):
    """
    Turn anything ComfyUI hands us (str / dict / File3D stream) into a real .glb path.
    """

    if isinstance(value, dict):
        for key in ("path", "file_path", "filepath", "model_path"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                resolved = _resolve_path_string(candidate)
                if os.path.exists(resolved):
                    return resolved

        raise RuntimeError("3D input file not found in dictionary input.")

    if isinstance(value, str):
        resolved = _resolve_path_string(value)
        if os.path.exists(resolved):
            return resolved

        raise RuntimeError("3D input file not found: " + resolved)

    obj = value
    fmt = str(getattr(obj, "format", None) or "glb")

    for attr in ("path", "file_path", "filepath", "source"):
        candidate = getattr(obj, attr, None)
        if isinstance(candidate, str):
            resolved = _resolve_path_string(candidate)
            if os.path.exists(resolved):
                return resolved

    data = None
    stream = getattr(obj, "source", None) or getattr(obj, "stream", None)

    if stream is not None and hasattr(stream, "read"):
        try:
            if hasattr(stream, "seek"):
                stream.seek(0)
            data = stream.read()
        except Exception:
            data = None

    if data is None:
        get_stream = getattr(obj, "get_stream", None)
        if callable(get_stream):
            try:
                with get_stream() as handle:
                    data = handle.read()
            except Exception:
                data = None

    if data is None:
        for meth in ("get_bytes", "to_bytes", "read"):
            method = getattr(obj, meth, None)
            if callable(method):
                try:
                    data = method()
                    break
                except Exception:
                    continue

    if data is None:
        target = os.path.join(out_dir, tag + "." + fmt)

        for meth in ("save", "save_to", "export", "write"):
            method = getattr(obj, meth, None)
            if callable(method):
                try:
                    method(target)
                    if os.path.exists(target):
                        return target
                except Exception:
                    continue

        raise RuntimeError("Could not read 3D input of type " + type(obj).__name__)

    if isinstance(data, str):
        resolved = _resolve_path_string(data)
        if os.path.exists(resolved):
            return resolved

        data = data.encode()

    target = os.path.join(out_dir, tag + "." + fmt)

    with open(target, "wb") as fh:
        fh.write(data)

    return target


class _BakedFile3D:
    """Minimal File3D-compatible object so Preview 3D / Load 3D nodes accept it."""

    def __init__(self, path, fmt="glb"):
        self.path = path
        self.file_path = path
        self.source = path
        self.format = fmt

    def get_stream(self):
        return open(self.path, "rb")

    def get_bytes(self):
        with open(self.path, "rb") as fh:
            return fh.read()

    def read(self):
        return self.get_bytes()

    def save_to(self, dest_path):
        import shutil
        shutil.copy2(self.path, dest_path)

    def save(self, dest_path):
        self.save_to(dest_path)


def _resolve_blender_binary(path):
    """
    Accept a full exe path OR a folder; always return the real executable.
    """

    candidate = (path or "").strip().strip('"').strip("'")

    if not candidate:
        return "blender"

    exe_name = "blender.exe" if sys.platform == "win32" else "blender"

    if os.path.isdir(candidate):
        direct = os.path.join(candidate, exe_name)

        if os.path.isfile(direct):
            return direct

        for entry in sorted(os.listdir(candidate)):
            nested = os.path.join(candidate, entry, exe_name)
            if os.path.isfile(nested):
                return nested

        raise RuntimeError("Could not find " + exe_name + " inside folder: " + candidate)

    if os.path.isfile(candidate):
        return candidate

    if sys.platform == "win32" and os.path.isfile(candidate + ".exe"):
        return candidate + ".exe"

    return candidate


def _parse_color(text, default):
    try:
        parts = [float(p) for p in str(text).replace(";", ",").split(",") if p.strip()]

        if len(parts) == 3:
            parts.append(1.0)

        if len(parts) != 4:
            return list(default)

        return parts
    except Exception:
        return list(default)


def _load_image_tensor(path):
    if Image is None or not path or not os.path.exists(path):
        return torch.zeros(1, 8, 8, 3, dtype=torch.float32)

    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    return torch.from_numpy(arr)[None, ...]


# ---------------------------------------------------------------------------
# Blender-side script
# ---------------------------------------------------------------------------

BLENDER_BAKE_SCRIPT = r'''
import json
import math
import os
import sys
import traceback
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def log(message):
    print("[BakeNode] " + str(message))


def configure_cycles_devices(s):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"

    warnings = []

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except Exception:
        return ["Cycles preferences unavailable; baking on CPU"]

    gpu_found = False

    if s.prefer_gpu_baking:
        for backend in ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI"):
            try:
                prefs.compute_device_type = backend
            except TypeError:
                continue

            try:
                prefs.refresh_devices()
            except Exception:
                try:
                    prefs.get_devices()
                except Exception:
                    pass

            if not [d for d in prefs.devices if d.type == backend]:
                continue

            for device in prefs.devices:
                device.use = device.type == backend or (s.hybrid_cpu_gpu_baking and device.type == "CPU")

            gpu_found = True
            break

    if not gpu_found:
        warnings.append("No GPU compute device found; falling back to CPU")

        try:
            prefs.compute_device_type = "NONE"
        except TypeError:
            pass

        for device in prefs.devices:
            device.use = True

    scene.cycles.device = "GPU" if gpu_found else "CPU"

    return warnings


def evaluated_world_bounds(mesh_object, dependency_graph):
    evaluated_object = mesh_object.evaluated_get(dependency_graph)
    world_corners = [evaluated_object.matrix_world @ Vector(c) for c in evaluated_object.bound_box]

    minimum = Vector(tuple(min(c[a] for c in world_corners) for a in range(3)))
    maximum = Vector(tuple(max(c[a] for c in world_corners) for a in range(3)))

    return minimum, maximum


def prepare_bake_only_settings(source_mesh, low_poly_mesh, s):
    maximum_mesh_dimension = max(low_poly_mesh.dimensions)

    if maximum_mesh_dimension <= 0.0:
        raise ValueError("Low poly has zero dimensions")

    dependency_graph = bpy.context.evaluated_depsgraph_get()

    source_minimum, source_maximum = evaluated_world_bounds(source_mesh, dependency_graph)
    low_minimum, low_maximum = evaluated_world_bounds(low_poly_mesh, dependency_graph)

    separation = max(
        max(0.0, low_minimum[i] - source_maximum[i], source_minimum[i] - low_maximum[i])
        for i in range(3)
    )

    if separation > maximum_mesh_dimension * 0.0001:
        raise ValueError("High and Low meshes do not overlap in world space")

    source_bvh = BVHTree.FromObject(source_mesh, dependency_graph)

    if source_bvh is None:
        raise ValueError("Could not read the high-poly surface")

    evaluated_low = low_poly_mesh.evaluated_get(dependency_graph)
    evaluated_mesh = evaluated_low.to_mesh()

    polygon_count = len(evaluated_mesh.polygons)

    if polygon_count == 0:
        evaluated_low.to_mesh_clear()
        raise ValueError("Low poly has no faces")

    sample_step = max(1, math.ceil(polygon_count / s.coverage_probe_sample_count))

    low_to_world = low_poly_mesh.matrix_world
    world_to_source = source_mesh.matrix_world.inverted()

    largest_gap = 0.0
    tested = 0

    try:
        for i in range(0, polygon_count, sample_step):
            world_pos = low_to_world @ evaluated_mesh.polygons[i].center
            nearest = source_bvh.find_nearest(world_to_source @ world_pos)

            if nearest is None:
                continue

            largest_gap = max(largest_gap, ((source_mesh.matrix_world @ nearest[0]) - world_pos).length)
            tested += 1
    finally:
        evaluated_low.to_mesh_clear()

    if tested == 0:
        raise ValueError("Could not compare High and Low surfaces")

    tight_reach = (s.tight_cage_extrusion_factor + s.tight_max_ray_distance_factor) * maximum_mesh_dimension

    if largest_gap <= tight_reach:
        return s

    expanded = SimpleNamespace(**vars(s))

    required = (largest_gap * 1.5 + maximum_mesh_dimension * 0.001) / maximum_mesh_dimension

    expanded.tight_max_ray_distance_factor = max(
        expanded.tight_max_ray_distance_factor,
        max(0.0, required - expanded.tight_cage_extrusion_factor)
    )

    expanded.far_max_ray_distance_factor = max(
        expanded.far_max_ray_distance_factor,
        required * 1.25
    )

    log("Adaptive reach expanded to cover gap " + str(largest_gap))

    return expanded


def probe_if_extended_rays_are_needed(source_mesh, low_poly_mesh, maximum_mesh_dimension, s):
    if s.fallback_bake_mode == "ALWAYS":
        return True

    if s.fallback_bake_mode == "NEVER":
        return False

    dependency_graph = bpy.context.evaluated_depsgraph_get()
    source_bvh = BVHTree.FromObject(source_mesh, dependency_graph)

    if source_bvh is None:
        return True

    tight_reach = (s.tight_cage_extrusion_factor + s.tight_max_ray_distance_factor) * maximum_mesh_dimension
    far_reach = (s.far_cage_extrusion_factor + s.far_max_ray_distance_factor) * maximum_mesh_dimension

    if far_reach <= tight_reach:
        return False

    evaluated_low = low_poly_mesh.evaluated_get(dependency_graph)
    evaluated_mesh = evaluated_low.to_mesh()

    polygon_count = len(evaluated_mesh.polygons)

    if polygon_count == 0:
        evaluated_low.to_mesh_clear()
        return False

    sample_step = max(1, math.ceil(polygon_count / s.coverage_probe_sample_count))

    low_to_world = low_poly_mesh.matrix_world
    world_to_source = source_mesh.matrix_world.inverted()

    recoverable = 0
    tested = 0

    try:
        for i in range(0, polygon_count, sample_step):
            world_pos = low_to_world @ evaluated_mesh.polygons[i].center
            nearest = source_bvh.find_nearest(world_to_source @ world_pos)

            if nearest is None:
                tested += 1
                continue

            distance = ((source_mesh.matrix_world @ nearest[0]) - world_pos).length

            if tight_reach < distance <= far_reach:
                recoverable += 1

            tested += 1
    finally:
        evaluated_low.to_mesh_clear()

    ratio = recoverable / max(1, tested)

    return recoverable >= s.fallback_minimum_recoverable_samples and ratio >= s.extended_ray_sample_ratio


def calculate_margin(resolution, s):
    if s.auto_bake_margin:
        return max(
            s.bake_margin_minimum_pixels,
            math.ceil(resolution / s.bake_margin_resolution_divisor)
        )

    return s.bake_margin_pixels


def allocate_transfer_image(name, size, semantic, s):
    existing = bpy.data.images.get(name)

    if existing is not None:
        bpy.data.images.remove(existing)

    # Only data maps (normal / roughness) use float buffers.
    # Color maps (base color / emission / AO) always use a clean
    # 8-bit sRGB buffer so there is never a mixed encoding in one texture.
    use_float = semantic in ("NORMAL", "ROUGHNESS")

    image = bpy.data.images.new(name, size, size, alpha=True, float_buffer=use_float)

    if semantic == "NORMAL":
        image.colorspace_settings.name = "Non-Color"
        n = s.neutral_normal_color
        image.generated_color = (n[0], n[1], n[2], 0.0)

    elif semantic == "ROUGHNESS":
        image.colorspace_settings.name = "Non-Color"
        r = float(getattr(s, "material_roughness", 0.9))
        image.generated_color = (r, r, r, 0.0)

    else:
        image.colorspace_settings.name = "sRGB"

        if semantic == "AO":
            image.generated_color = (1.0, 1.0, 1.0, 0.0)
        else:
            image.generated_color = (0.0, 0.0, 0.0, 0.0)

    return image


def _pixel_array(image):
    buffer = np.empty(image.size[0] * image.size[1] * 4, dtype=np.float32)
    image.pixels.foreach_get(buffer)
    return buffer.reshape(-1, 4)


def _write_pixels(image, array):
    image.pixels.foreach_set(np.ascontiguousarray(array, dtype=np.float32).ravel())
    image.update()


def _linear_to_srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _covered_mask(array, semantic, s):
    covered = array[:, 3] >= s.coverage_alpha_threshold

    if semantic == "NORMAL":
        vector = array[:, :3] * 2.0 - 1.0
        squared = (vector * vector).sum(axis=1)

        low = (1.0 - s.normal_length_tolerance) ** 2
        high = (1.0 + s.normal_length_tolerance) ** 2

        covered = covered & (squared >= low) & (squared <= high)

    return covered


def classify_uncovered_texels(image, semantic, s):
    return ~_covered_mask(_pixel_array(image), semantic, s)


def finalize_texels(image, fallback_image, missing, semantic, s):
    if not missing.any():
        return 0, 0

    main_array = _pixel_array(image)

    taken = 0
    remaining = missing

    if fallback_image is not None:
        fallback_array = _pixel_array(fallback_image)

        take = missing & _covered_mask(fallback_array, semantic, s)

        main_array[take] = fallback_array[take]

        taken = int(take.sum())
        remaining = missing & ~take

    filled = 0

    if remaining.any():
        if semantic == "NORMAL":
            n = s.neutral_normal_color
            fill = np.array([n[0], n[1], n[2], 1.0], dtype=np.float32)

        elif semantic == "ROUGHNESS":
            r = float(getattr(s, "material_roughness", 0.9))
            fill = np.array([r, r, r, 1.0], dtype=np.float32)

        elif semantic == "AO":
            fill = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        elif semantic == "EMISSION":
            fill = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        else:
            # Base color buffer is always 8-bit sRGB, so the fill
            # value must always be sRGB encoded too.
            b = s.material_base_color
            fill = np.array(
                [
                    _linear_to_srgb(b[0]),
                    _linear_to_srgb(b[1]),
                    _linear_to_srgb(b[2]),
                    1.0
                ],
                dtype=np.float32
            )

        main_array[remaining] = fill
        filled = int(remaining.sum())

    _write_pixels(image, main_array)

    return taken, filled


def boost_base_color_brightness(image, exponent):
    array = _pixel_array(image)
    array[:, :3] = np.clip(array[:, :3], 0.0, 1.0) ** exponent
    _write_pixels(image, array)


def suppress_metallic(mesh_object):
    """
    Metallic > 0 makes a Diffuse-Color bake go black.
    Temporarily zero it on the high poly and remember how to restore it.
    """

    restore = []

    for slot in mesh_object.material_slots:
        mat = slot.material

        if mat is None or not mat.use_nodes:
            continue

        for node in mat.node_tree.nodes:
            if node.bl_idname != "ShaderNodeBsdfPrincipled":
                continue

            inp = node.inputs.get("Metallic")

            if inp is None:
                continue

            if inp.is_linked:
                from_socket = inp.links[0].from_socket
                mat.node_tree.links.remove(inp.links[0])
                restore.append((mat, inp, from_socket, inp.default_value))
            else:
                restore.append((mat, inp, None, inp.default_value))

            inp.default_value = 0.0

    return restore


def restore_metallic(restore):
    for mat, inp, from_socket, original_value in restore:
        inp.default_value = original_value

        if from_socket is not None:
            mat.node_tree.links.new(from_socket, inp)


def _try_hide(obj, state):
    try:
        obj.hide_set(state)
    except Exception:
        pass


def select_pair(source_mesh, low_poly_mesh):
    if bpy.context.object is not None and bpy.context.object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    for obj in bpy.data.objects:
        try:
            obj.select_set(False)
        except Exception:
            pass

    for obj in (source_mesh, low_poly_mesh):
        obj.hide_render = False
        _try_hide(obj, False)
        obj.select_set(True)

    bpy.context.view_layer.objects.active = low_poly_mesh


def activate_bake_destination(material, node):
    tree = material.node_tree

    for n in tree.nodes:
        n.select = False

    node.select = True
    tree.nodes.active = node


def run_bake_pass(source_mesh, low_poly_mesh, material, node, bake_type, cage, ray, margin, samples):
    select_pair(source_mesh, low_poly_mesh)
    activate_bake_destination(material, node)

    bpy.context.scene.cycles.samples = samples

    args = {
        "type": bake_type,
        "use_selected_to_active": True,
        "use_clear": False,
        "cage_extrusion": cage,
        "max_ray_distance": ray,
        "margin": margin,
        "margin_type": "EXTEND",
    }

    if bake_type == "DIFFUSE":
        args["pass_filter"] = {"COLOR"}
    elif bake_type == "NORMAL":
        args["normal_space"] = "TANGENT"

    bpy.ops.object.bake(**args)


def build_transfer_material(
    base_name,
    normal_image,
    base_color_image,
    roughness_image,
    emission_image,
    ao_image,
    s
):
    material = bpy.data.materials.new(base_name + "_Baked_Mat")
    material.use_nodes = True

    tree = material.node_tree

    shader = next(n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled")

    shader.inputs["Base Color"].default_value = s.material_base_color
    shader.inputs["Roughness"].default_value = s.material_roughness

    spec = "Specular IOR Level" if "Specular IOR Level" in shader.inputs else "Specular"
    shader.inputs[spec].default_value = s.material_specular_level

    nodes = {}

    if normal_image is not None:
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = normal_image

        nmap = tree.nodes.new("ShaderNodeNormalMap")

        tree.links.new(tex.outputs["Color"], nmap.inputs["Color"])
        tree.links.new(nmap.outputs["Normal"], shader.inputs["Normal"])

        nodes["NORMAL"] = tex

    if base_color_image is not None:
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = base_color_image

        tree.links.new(tex.outputs["Color"], shader.inputs["Base Color"])

        nodes["BASE_COLOR"] = tex

    if roughness_image is not None:
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = roughness_image

        tree.links.new(tex.outputs["Color"], shader.inputs["Roughness"])

        nodes["ROUGHNESS"] = tex

    if emission_image is not None:
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = emission_image

        if "Emission Color" in shader.inputs:
            tree.links.new(tex.outputs["Color"], shader.inputs["Emission Color"])

            if "Emission Strength" in shader.inputs:
                shader.inputs["Emission Strength"].default_value = 1.0

        elif "Emission" in shader.inputs:
            tree.links.new(tex.outputs["Color"], shader.inputs["Emission"])

        nodes["EMISSION"] = tex

    if ao_image is not None:
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.image = ao_image

        # AO is delivered as an image map only (Principled has no AO input).
        nodes["AO"] = tex

    return material, nodes


def transfer_map(source_mesh, low_poly_mesh, material, node, image, bake_type,
                 strategy, max_dim, s, samples=None):
    semantic_map = {
        "NORMAL": "NORMAL",
        "DIFFUSE": "BASE_COLOR",
        "ROUGHNESS": "ROUGHNESS",
        "EMIT": "EMISSION",
        "AO": "AO",
    }

    semantic = semantic_map.get(bake_type, "BASE_COLOR")

    if samples is None:
        samples = s.bake_sample_count

    margin = calculate_margin(image.size[0], s)

    close_cage = s.tight_cage_extrusion_factor * max_dim
    close_ray = s.tight_max_ray_distance_factor * max_dim

    far_cage = s.far_cage_extrusion_factor * max_dim
    far_ray = s.far_max_ray_distance_factor * max_dim

    first_cage, first_ray = (far_cage, far_ray) if strategy == "EXTENDED_SINGLE" else (close_cage, close_ray)

    run_bake_pass(
        source_mesh,
        low_poly_mesh,
        material,
        node,
        bake_type,
        first_cage,
        first_ray,
        margin,
        samples
    )

    missing = classify_uncovered_texels(image, semantic, s)

    fallback_image = None
    fallback_node = None

    if strategy == "TWO_PASS" and missing.any():
        fallback_image = allocate_transfer_image(
            image.name + "_Fallback",
            image.size[0],
            semantic,
            s
        )

        fallback_node = material.node_tree.nodes.new("ShaderNodeTexImage")
        fallback_node.image = fallback_image

        run_bake_pass(
            source_mesh,
            low_poly_mesh,
            material,
            fallback_node,
            bake_type,
            far_cage,
            far_ray,
            margin,
            samples
        )

    try:
        merged, filled = finalize_texels(image, fallback_image, missing, semantic, s)
        log(semantic + ": merged=" + str(merged) + " filled=" + str(filled))
    finally:
        if fallback_node is not None:
            material.node_tree.nodes.remove(fallback_node)

        if fallback_image is not None:
            bpy.data.images.remove(fallback_image)


def import_glb_as_single_mesh(path, tag):
    before = set(bpy.data.objects)

    bpy.ops.import_scene.gltf(filepath=path)

    new_meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]

    if not new_meshes:
        raise ValueError("No meshes found in " + path)

    for o in bpy.data.objects:
        try:
            o.select_set(False)
        except Exception:
            pass

    for o in new_meshes:
        o.select_set(True)

    bpy.context.view_layer.objects.active = new_meshes[0]

    if len(new_meshes) > 1:
        bpy.ops.object.join()

    result = bpy.context.view_layer.objects.active
    result.name = tag

    return result


def save_image(image, directory, filename):
    path = os.path.join(directory, filename)

    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()

    return path


def main():
    argv = sys.argv
    config_path = argv[argv.index("--") + 1]

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    s = SimpleNamespace(**cfg["settings"])

    s.neutral_normal_color = tuple(s.neutral_normal_color)
    s.material_base_color = tuple(s.material_base_color)

    high = import_glb_as_single_mesh(cfg["high_poly"], "BakeHigh")
    low = import_glb_as_single_mesh(cfg["low_poly"], "BakeLow")

    wants_normal = bool(getattr(s, "bake_normal_detail", True))
    wants_base_color = bool(getattr(s, "copy_base_color_texture", True))
    wants_roughness = bool(getattr(s, "bake_roughness_texture", True))
    wants_emission = bool(getattr(s, "bake_emission_texture", True))
    wants_ao = bool(getattr(s, "bake_ao_texture", True))

    if not (wants_normal or wants_base_color or wants_roughness or wants_emission or wants_ao):
        raise ValueError("Enable at least one bake/map output.")

    if not low.data.uv_layers:
        raise ValueError("Low poly needs a UV map")

    warnings = configure_cycles_devices(s)

    settings = prepare_bake_only_settings(high, low, s)

    max_dim = max(low.dimensions)

    if settings.fallback_bake_mode == "ALWAYS":
        strategy = "TWO_PASS"
    elif settings.fallback_bake_mode == "NEVER":
        strategy = "TIGHT_SINGLE"
    else:
        strategy = "EXTENDED_SINGLE" if probe_if_extended_rays_are_needed(high, low, max_dim, settings) else "TIGHT_SINGLE"

    base_name = low.name[:40]

    normal_image = None
    base_color_image = None
    roughness_image = None
    emission_image = None
    ao_image = None

    if wants_normal:
        normal_image = allocate_transfer_image(
            base_name + "_Normal",
            settings.normal_map_resolution,
            "NORMAL",
            settings
        )

    if wants_base_color:
        base_color_image = allocate_transfer_image(
            base_name + "_BaseColor",
            settings.base_color_resolution,
            "BASE_COLOR",
            settings
        )

    if wants_roughness:
        roughness_image = allocate_transfer_image(
            base_name + "_Roughness",
            getattr(settings, "roughness_resolution", settings.base_color_resolution),
            "ROUGHNESS",
            settings
        )

    if wants_emission:
        emission_image = allocate_transfer_image(
            base_name + "_Emission",
            getattr(settings, "emission_resolution", settings.base_color_resolution),
            "EMISSION",
            settings
        )

    if wants_ao:
        ao_image = allocate_transfer_image(
            base_name + "_AO",
            getattr(settings, "ao_resolution", settings.base_color_resolution),
            "AO",
            settings
        )

    material, nodes = build_transfer_material(
        base_name,
        normal_image,
        base_color_image,
        roughness_image,
        emission_image,
        ao_image,
        settings
    )

    low.data.materials.clear()
    low.data.materials.append(material)

    if normal_image is not None:
        log("Baking normal detail...")
        transfer_map(
            high,
            low,
            material,
            nodes["NORMAL"],
            normal_image,
            "NORMAL",
            strategy,
            max_dim,
            settings
        )

    if base_color_image is not None:
        log("Baking base color...")

        # Metallic surfaces bake to black in a Diffuse-Color pass.
        # Zero metallic on the high poly only for this bake, then restore.
        metallic_restore = suppress_metallic(high)

        try:
            transfer_map(
                high,
                low,
                material,
                nodes["BASE_COLOR"],
                base_color_image,
                "DIFFUSE",
                strategy,
                max_dim,
                settings
            )
        finally:
            restore_metallic(metallic_restore)

        if settings.brighter:
            boost_base_color_brightness(base_color_image, 1.0 / 1.2)

    if roughness_image is not None:
        log("Baking roughness...")
        transfer_map(
            high,
            low,
            material,
            nodes["ROUGHNESS"],
            roughness_image,
            "ROUGHNESS",
            strategy,
            max_dim,
            settings
        )

    if emission_image is not None:
        log("Baking emission...")
        transfer_map(
            high,
            low,
            material,
            nodes["EMISSION"],
            emission_image,
            "EMIT",
            strategy,
            max_dim,
            settings
        )

    if ao_image is not None:
        log("Baking AO...")
        transfer_map(
            high,
            low,
            material,
            nodes["AO"],
            ao_image,
            "AO",
            strategy,
            max_dim,
            settings,
            samples=max(32, settings.bake_sample_count)
        )

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    result = {"warnings": warnings}

    if normal_image is not None:
        result["normal_png"] = save_image(normal_image, out_dir, base_name + "_normal.png")

    if base_color_image is not None:
        result["base_color_png"] = save_image(base_color_image, out_dir, base_name + "_basecolor.png")

    if roughness_image is not None:
        result["roughness_png"] = save_image(roughness_image, out_dir, base_name + "_roughness.png")

    if emission_image is not None:
        result["emission_png"] = save_image(emission_image, out_dir, base_name + "_emission.png")

    if ao_image is not None:
        result["ao_png"] = save_image(ao_image, out_dir, base_name + "_ao.png")

    for o in bpy.data.objects:
        try:
            o.select_set(False)
        except Exception:
            pass

    low.select_set(True)
    bpy.context.view_layer.objects.active = low

    bpy.ops.export_scene.gltf(
        filepath=cfg["glb_out"],
        use_selection=True,
        export_format="GLB",
        export_yup=True,
        export_tangents=True,
    )

    result["glb"] = cfg["glb_out"]
    result["status"] = "FINISHED"

    with open(cfg["result_json"], "w", encoding="utf-8") as fh:
        json.dump(result, fh)

    log("DONE")


try:
    main()
except Exception:
    traceback.print_exc()

    try:
        with open(sys.argv[sys.argv.index("--") + 1], "r", encoding="utf-8") as fh:
            cfg = json.load(fh)

        with open(cfg["result_json"], "w", encoding="utf-8") as fh:
            json.dump({"status": "FAILED", "error": traceback.format_exc()}, fh)
    except Exception:
        pass

    sys.exit(1)
'''


class LODTailorBakeForger:
    """ComfyUI node that runs the High-to-Active bake in a separate Blender process."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_poly": ("FILE_3D_GLB",),
                "low_poly": ("FILE_3D_GLB",),

                "blender_path": ("STRING", {
                    "default": "blender",
                    "tooltip": "Full path to your Blender executable (or 'blender' if on PATH)"
                }),

                "prefer_gpu_baking": ("BOOLEAN", {"default": True}),
                "hybrid_cpu_gpu_baking": ("BOOLEAN", {"default": True}),

                "bake_sample_count": ("INT", {"default": 1, "min": 1, "max": 4096}),

                "bake_normal_detail": ("BOOLEAN", {"default": True}),
                "normal_map_resolution": ("INT", {"default": 8192, "min": 16, "max": 32768}),

                "copy_base_color_texture": ("BOOLEAN", {"default": True}),
                "base_color_resolution": ("INT", {"default": 8192, "min": 16, "max": 32768}),

                "bake_roughness_texture": ("BOOLEAN", {"default": True}),
                "roughness_resolution": ("INT", {"default": 8192, "min": 16, "max": 32768}),

                "bake_emission_texture": ("BOOLEAN", {"default": True}),
                "emission_resolution": ("INT", {"default": 8192, "min": 16, "max": 32768}),

                "bake_ao_texture": ("BOOLEAN", {"default": True}),
                "ao_resolution": ("INT", {"default": 8192, "min": 16, "max": 32768}),

                "brighter": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Applies a mild uniform brightness boost to the baked base color map"
                }),

                "auto_bake_margin": ("BOOLEAN", {"default": True}),
                "bake_margin_pixels": ("INT", {"default": 16, "min": 0, "max": 1024}),
                "bake_margin_minimum_pixels": ("INT", {"default": 16, "min": 0, "max": 1024}),
                "bake_margin_resolution_divisor": ("INT", {"default": 256, "min": 1, "max": 4096}),

                "fallback_bake_mode": (["AUTO", "ALWAYS", "NEVER"], {"default": "AUTO"}),

                "coverage_probe_sample_count": ("INT", {"default": 4096, "min": 64, "max": 100000}),
                "extended_ray_sample_ratio": ("FLOAT", {"default": 0.005, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "fallback_minimum_recoverable_samples": ("INT", {"default": 8, "min": 1, "max": 10000}),

                "tight_cage_extrusion_factor": ("FLOAT", {"default": 0.0019, "min": 0.0, "step": 0.0001}),
                "tight_max_ray_distance_factor": ("FLOAT", {"default": 0.029, "min": 0.0, "step": 0.0001}),

                "far_cage_extrusion_factor": ("FLOAT", {"default": 0.020, "min": 0.0, "step": 0.0001}),
                "far_max_ray_distance_factor": ("FLOAT", {"default": 0.092, "min": 0.0, "step": 0.0001}),

                "coverage_alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "normal_length_tolerance": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95}),

                "neutral_normal_color": ("STRING", {"default": "0.5,0.5,1.0,1.0"}),
                "material_base_color": ("STRING", {"default": "0.8,0.8,0.8,1.0"}),

                "material_roughness": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "material_specular_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = (
        "FILE_3D_GLB",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "IMAGE",
        "STRING",
    )

    RETURN_NAMES = (
        "baked_glb",
        "base_color",
        "normal",
        "roughness",
        "emission",
        "ao",
        "info",
    )

    FUNCTION = "bake"
    CATEGORY = "bake"

    def bake(
        self,
        high_poly,
        low_poly,
        blender_path,
        neutral_normal_color,
        material_base_color,
        **kw
    ):
        out_dir = tempfile.mkdtemp(prefix="comfy_bake_hta_")

        high_path = _extract_3d_path(high_poly, out_dir, "input_high")
        low_path = _extract_3d_path(low_poly, out_dir, "input_low")

        for p in (high_path, low_path):
            if not os.path.exists(p):
                raise RuntimeError("3D input file not found: " + str(p))

        glb_out = os.path.join(out_dir, "baked_low.glb")
        result_json = os.path.join(out_dir, "result.json")

        script_path = os.path.join(out_dir, "bake_script.py")
        config_path = os.path.join(out_dir, "config.json")

        settings = dict(kw)

        settings["neutral_normal_color"] = _parse_color(neutral_normal_color, (0.5, 0.5, 1.0, 1.0))
        settings["material_base_color"] = _parse_color(material_base_color, (0.8, 0.8, 0.8, 1.0))

        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(BLENDER_BAKE_SCRIPT)

        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "high_poly": high_path,
                    "low_poly": low_path,
                    "out_dir": out_dir,
                    "glb_out": glb_out,
                    "result_json": result_json,
                    "settings": settings,
                },
                fh
            )

        blender_bin = _resolve_blender_binary(blender_path)

        creation_flags = 0

        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        try:
            process = subprocess.run(
                [blender_bin, "-b", "-P", script_path, "--", config_path],
                capture_output=True,
                text=True,
                creationflags=creation_flags,
            )
        except (PermissionError, FileNotFoundError, OSError) as exc:
            raise RuntimeError(
                "Could not launch Blender at '" + str(blender_bin) + "': " + str(exc) +
                " — set blender_path to the full path of blender.exe"
            )

        result = {}

        if os.path.exists(result_json):
            with open(result_json, "r", encoding="utf-8") as fh:
                result = json.load(fh)

        if process.returncode != 0 or result.get("status") != "FINISHED":
            tail = (process.stdout or "")[-2000:] + (process.stderr or "")[-2000:]
            raise RuntimeError("Blender bake failed: " + str(result.get("error", tail)))

        base_tensor = _load_image_tensor(result.get("base_color_png", ""))
        normal_tensor = _load_image_tensor(result.get("normal_png", ""))
        roughness_tensor = _load_image_tensor(result.get("roughness_png", ""))
        emission_tensor = _load_image_tensor(result.get("emission_png", ""))
        ao_tensor = _load_image_tensor(result.get("ao_png", ""))

        file_3d = _BakedFile3D(result["glb"], "glb")

        info = json.dumps(result, indent=2)

        return (
            file_3d,
            base_tensor,
            normal_tensor,
            roughness_tensor,
            emission_tensor,
            ao_tensor,
            info,
        )


NODE_CLASS_MAPPINGS = {
    "LODTailorBakeForger": LODTailorBakeForger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LODTailorBakeForger": "LODTailor: Bake Forger",
}