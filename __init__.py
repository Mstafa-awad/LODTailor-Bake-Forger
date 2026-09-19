"""ComfyUI custom node: Bake High to Active Low (optimized)"""
import json
import os
import subprocess
import sys
import tempfile

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


def _empty_image_tensor():
    return torch.zeros(1, 8, 8, 3, dtype=torch.float32)


def _load_image_tensor(path):
    if Image is None or not path or not os.path.exists(path):
        return _empty_image_tensor()

    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32)

    arr /= 255.0
    return torch.from_numpy(arr)[None, ...]


BLENDER_BAKE_SCRIPT = r'''
import json
import math
import os
import sys
import gc
import traceback
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

try:
    bpy.context.preferences.edit.use_global_undo = False
except Exception:
    pass

try:
    bpy.context.preferences.filepaths.use_auto_save_temporary_files = False
except Exception:
    pass

try:
    bpy.context.preferences.filepaths.save_version = 0
except Exception:
    pass


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


def prepare_settings_and_strategy(source_mesh, low_poly_mesh, s):
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
    source_to_world = source_mesh.matrix_world

    largest_gap = 0.0
    tested_found = 0
    total_tested = 0
    distances = []

    try:
        for i in range(0, polygon_count, sample_step):
            total_tested += 1
            world_pos = low_to_world @ evaluated_mesh.polygons[i].center
            nearest = source_bvh.find_nearest(world_to_source @ world_pos)

            if nearest is None:
                continue

            distance = ((source_to_world @ nearest[0]) - world_pos).length
            distances.append(distance)

            if distance > largest_gap:
                largest_gap = distance

            tested_found += 1
    finally:
        evaluated_low.to_mesh_clear()

    if tested_found == 0:
        raise ValueError("Could not compare High and Low surfaces")

    tight_reach = (s.tight_cage_extrusion_factor + s.tight_max_ray_distance_factor) * maximum_mesh_dimension

    settings = s
    if largest_gap > tight_reach:
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

        settings = expanded
        log("Adaptive reach expanded to cover gap " + str(largest_gap))

    if settings.fallback_bake_mode == "ALWAYS":
        strategy = "TWO_PASS"
    elif settings.fallback_bake_mode == "NEVER":
        strategy = "TIGHT_SINGLE"
    else:
        tight_reach = (settings.tight_cage_extrusion_factor + settings.tight_max_ray_distance_factor) * maximum_mesh_dimension
        far_reach = (settings.far_cage_extrusion_factor + settings.far_max_ray_distance_factor) * maximum_mesh_dimension

        if far_reach <= tight_reach:
            strategy = "TIGHT_SINGLE"
        else:
            recoverable = 0
            for d in distances:
                if tight_reach < d <= far_reach:
                    recoverable += 1

            ratio = recoverable / max(1, total_tested)
            if recoverable >= settings.fallback_minimum_recoverable_samples and ratio >= settings.extended_ray_sample_ratio:
                strategy = "EXTENDED_SINGLE"
            else:
                strategy = "TIGHT_SINGLE"

    return settings, strategy, maximum_mesh_dimension


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

    use_float = semantic in ("NORMAL", "ROUGHNESS", "METALLIC")
    image = bpy.data.images.new(name, size, size, alpha=True, float_buffer=use_float)

    if semantic == "NORMAL":
        image.colorspace_settings.name = "Non-Color"
        n = s.neutral_normal_color
        image.generated_color = (n[0], n[1], n[2], 0.0)
    elif semantic == "ROUGHNESS":
        image.colorspace_settings.name = "Non-Color"
        r = float(getattr(s, "material_roughness", 0.9))
        image.generated_color = (r, r, r, 0.0)
    elif semantic == "METALLIC":
        image.colorspace_settings.name = "Non-Color"
        m = float(getattr(s, "material_metallic", 0.0))
        image.generated_color = (m, m, m, 0.0)
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
    if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array, dtype=np.float32)
    image.pixels.foreach_set(array.ravel())
    image.update()


def _linear_to_srgb(c):
    return c * 12.92 if c <= 0.0031308 else 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _covered_mask_from_array(array, semantic, s):
    covered = array[:, 3] >= float(s.coverage_alpha_threshold)

    if semantic != "NORMAL":
        return covered

    count = array.shape[0]
    if count == 0:
        return covered

    tolerance = float(s.normal_length_tolerance)
    low = (1.0 - tolerance) ** 2
    high = (1.0 + tolerance) ** 2

    rgb = array[:, :3]
    chunk_size = 1048576

    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        block = rgb[start:end]
        if block.size == 0:
            continue

        v = block * 2.0 - 1.0
        squared = np.einsum("ij,ij->i", v, v)
        covered[start:end] &= (squared >= low) & (squared <= high)

    return covered


def _fill_value(semantic, s):
    if semantic == "NORMAL":
        n = tuple(s.neutral_normal_color)
        return np.array([n[0], n[1], n[2], 1.0], dtype=np.float32)

    if semantic == "ROUGHNESS":
        r = float(getattr(s, "material_roughness", 0.9))
        return np.array([r, r, r, 1.0], dtype=np.float32)

    if semantic == "METALLIC":
        m = float(getattr(s, "material_metallic", 0.0))
        return np.array([m, m, m, 1.0], dtype=np.float32)

    if semantic == "AO":
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    if semantic == "EMISSION":
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    b = tuple(s.material_base_color)
    return np.array(
        [
            _linear_to_srgb(b[0]),
            _linear_to_srgb(b[1]),
            _linear_to_srgb(b[2]),
            1.0,
        ],
        dtype=np.float32
    )


def fill_missing_in_array(array, missing, semantic, s):
    if not missing.any():
        return 0

    array[missing] = _fill_value(semantic, s)
    return int(missing.sum())


def _boost_base_color_array(array, exponent):
    rgb = array[:, :3]
    np.clip(rgb, 0.0, 1.0, out=rgb)
    np.power(rgb, exponent, out=rgb)


def finalize_with_missing(image, fallback_image, missing, semantic, s, array_post_process=None):
    if not missing.any():
        return 0, 0

    main_array = _pixel_array(image)
    taken = 0
    remaining = missing

    if fallback_image is not None:
        fallback_array = _pixel_array(fallback_image)
        fallback_covered = _covered_mask_from_array(fallback_array, semantic, s)

        take = missing & fallback_covered
        taken = int(take.sum())

        if taken > 0:
            main_array[take] = fallback_array[take]

        remaining = missing & ~take

        del fallback_array
        del fallback_covered
        del take
        gc.collect()

    filled = 0
    if remaining.any():
        main_array[remaining] = _fill_value(semantic, s)
        filled = int(remaining.sum())

    if array_post_process is not None:
        array_post_process(main_array)

    _write_pixels(image, main_array)

    del main_array
    del remaining
    gc.collect()

    return taken, filled


def suppress_metallic(mesh_object):
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


def swap_metallic_to_emission(mesh_object):
    restore_records = []

    for slot in mesh_object.material_slots:
        mat = slot.material
        if mat is None or not mat.use_nodes:
            continue

        tree = mat.node_tree

        principled = None
        output_node = None

        for n in tree.nodes:
            if n.bl_idname == "ShaderNodeBsdfPrincipled" and principled is None:
                principled = n
            if n.bl_idname == "ShaderNodeOutputMaterial" and output_node is None:
                output_node = n

        if principled is None or output_node is None:
            continue

        metallic_inp = principled.inputs.get("Metallic")
        if metallic_inp is None:
            continue

        output_surface = output_node.inputs.get("Surface")
        if output_surface is None:
            continue

        original_surface_link = None
        if output_surface.is_linked:
            original_surface_link = output_surface.links[0].from_socket

        metallic_from_socket = None
        metallic_default = metallic_inp.default_value

        if metallic_inp.is_linked:
            metallic_from_socket = metallic_inp.links[0].from_socket
            tree.links.remove(metallic_inp.links[0])

        emission = tree.nodes.new("ShaderNodeEmission")
        emission.name = "_BakeMetallicEmit"
        emission.location = (principled.location[0] + 300, principled.location[1] + 300)

        if metallic_from_socket is not None:
            tree.links.new(metallic_from_socket, emission.inputs["Color"])
        else:
            rgb = tree.nodes.new("ShaderNodeRGB")
            rgb.name = "_BakeMetallicRGB"
            rgb.outputs[0].default_value = (
                metallic_default,
                metallic_default,
                metallic_default,
                1.0
            )
            tree.links.new(rgb.outputs[0], emission.inputs["Color"])

        if original_surface_link is not None:
            tree.links.remove(output_surface.links[0])

        tree.links.new(emission.outputs["Emission"], output_surface)

        restore_records.append({
            "material": mat,
            "tree": tree,
            "principled": principled,
            "output_node": output_node,
            "metallic_inp": metallic_inp,
            "metallic_from_socket": metallic_from_socket,
            "metallic_default": metallic_default,
            "original_surface_link": original_surface_link,
            "emission_node": emission,
        })

    return restore_records


def restore_emission_swap(records):
    for rec in records:
        try:
            tree = rec["tree"]
            emission = rec["emission_node"]
            output_node = rec["output_node"]
            output_surface = output_node.inputs.get("Surface")

            for inp in emission.inputs:
                for link in list(inp.links):
                    tree.links.remove(link)

            tree.nodes.remove(emission)

            for n in list(tree.nodes):
                if n.name.startswith("_BakeMetallicRGB"):
                    for out in n.outputs:
                        for link in list(out.links):
                            tree.links.remove(link)
                    tree.nodes.remove(n)

            if output_surface is not None:
                for link in list(output_surface.links):
                    tree.links.remove(link)

                if rec["original_surface_link"] is not None:
                    tree.links.new(rec["original_surface_link"], output_surface)

            metallic_inp = rec["metallic_inp"]
            metallic_inp.default_value = rec["metallic_default"]

            if rec["metallic_from_socket"] is not None:
                tree.links.new(rec["metallic_from_socket"], metallic_inp)
        except Exception:
            pass


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


def set_material_defaults(shader, s):
    shader.inputs["Base Color"].default_value = tuple(s.material_base_color)
    shader.inputs["Roughness"].default_value = float(s.material_roughness)
    shader.inputs["Metallic"].default_value = float(getattr(s, "material_metallic", 0.0))

    spec_name = "Specular IOR Level" if "Specular IOR Level" in shader.inputs else "Specular"
    if spec_name in shader.inputs:
        shader.inputs[spec_name].default_value = float(s.material_specular_level)


def build_bake_material(base_name, s):
    material = bpy.data.materials.new(base_name + "_Bake_Mat")
    material.use_nodes = True
    tree = material.node_tree

    shader = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if shader is None:
        shader = tree.nodes.new("ShaderNodeBsdfPrincipled")

    output = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")

    if not output.inputs["Surface"].is_linked:
        tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    set_material_defaults(shader, s)
    return material


def save_image(image, directory, filename):
    path = os.path.join(directory, filename)
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    return path


def process_map(
    source_mesh,
    low_poly_mesh,
    material,
    bake_type,
    semantic,
    image_name,
    resolution,
    strategy,
    max_dim,
    s,
    out_dir,
    filename,
    samples=None,
    array_post_process=None
):
    image = allocate_transfer_image(image_name, resolution, semantic, s)

    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image

    margin = calculate_margin(resolution, s)

    close_cage = s.tight_cage_extrusion_factor * max_dim
    close_ray = s.tight_max_ray_distance_factor * max_dim
    far_cage = s.far_cage_extrusion_factor * max_dim
    far_ray = s.far_max_ray_distance_factor * max_dim

    if samples is None:
        samples = s.bake_sample_count

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

    main_array = _pixel_array(image)
    missing = ~_covered_mask_from_array(main_array, semantic, s)
    has_missing = bool(missing.any())

    taken = 0
    filled = 0

    if strategy == "TWO_PASS" and has_missing:
        del main_array
        gc.collect()

        fallback_image = allocate_transfer_image(image_name + "_Fallback", resolution, semantic, s)
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
            taken, filled = finalize_with_missing(
                image,
                fallback_image,
                missing,
                semantic,
                s,
                array_post_process
            )
        finally:
            material.node_tree.nodes.remove(fallback_node)
            bpy.data.images.remove(fallback_image)
            gc.collect()
    else:
        if has_missing:
            filled = fill_missing_in_array(main_array, missing, semantic, s)

        if array_post_process is not None:
            array_post_process(main_array)

        if has_missing or array_post_process is not None:
            _write_pixels(image, main_array)

        del main_array
        gc.collect()

    del missing
    gc.collect()

    path = save_image(image, out_dir, filename)

    material.node_tree.nodes.remove(node)
    bpy.data.images.remove(image)
    gc.collect()

    log(semantic + ": merged=" + str(taken) + " filled=" + str(filled))
    return path


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


def purge_orphan_datablocks():
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images):
        for item in list(collection):
            if item.users == 0:
                try:
                    collection.remove(item)
                except Exception:
                    pass


def build_export_material(base_name, texture_paths, s):
    material = bpy.data.materials.new(base_name + "_Export_Mat")
    material.use_nodes = True
    tree = material.node_tree

    shader = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if shader is None:
        shader = tree.nodes.new("ShaderNodeBsdfPrincipled")

    output = next((n for n in tree.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if output is None:
        output = tree.nodes.new("ShaderNodeOutputMaterial")

    if not output.inputs["Surface"].is_linked:
        tree.links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    set_material_defaults(shader, s)

    def make_image_node(semantic, path):
        if not path or not os.path.exists(path):
            return None

        image = bpy.data.images.load(path, check_existing=False)

        if semantic in ("NORMAL", "ROUGHNESS", "METALLIC"):
            image.colorspace_settings.name = "Non-Color"
        else:
            image.colorspace_settings.name = "sRGB"

        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        return node

    normal_node = make_image_node("NORMAL", texture_paths.get("NORMAL"))
    if normal_node is not None:
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        tree.links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], shader.inputs["Normal"])

    base_node = make_image_node("BASE_COLOR", texture_paths.get("BASE_COLOR"))
    if base_node is not None:
        tree.links.new(base_node.outputs["Color"], shader.inputs["Base Color"])

    rough_node = make_image_node("ROUGHNESS", texture_paths.get("ROUGHNESS"))
    if rough_node is not None:
        tree.links.new(rough_node.outputs["Color"], shader.inputs["Roughness"])

    metallic_node = make_image_node("METALLIC", texture_paths.get("METALLIC"))
    if metallic_node is not None:
        tree.links.new(metallic_node.outputs["Color"], shader.inputs["Metallic"])

    emission_node = make_image_node("EMISSION", texture_paths.get("EMISSION"))
    if emission_node is not None:
        if "Emission Color" in shader.inputs:
            tree.links.new(emission_node.outputs["Color"], shader.inputs["Emission Color"])
            if "Emission Strength" in shader.inputs:
                shader.inputs["Emission Strength"].default_value = 1.0
        elif "Emission" in shader.inputs:
            tree.links.new(emission_node.outputs["Color"], shader.inputs["Emission"])

    # AO remains available as a texture node, but is not linked into Principled.
    make_image_node("AO", texture_paths.get("AO"))

    return material


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

    purge_orphan_datablocks()

    wants_normal = bool(getattr(s, "bake_normal_detail", True))
    wants_base_color = bool(getattr(s, "copy_base_color_texture", True))
    wants_roughness = bool(getattr(s, "bake_roughness_texture", True))
    wants_metallic = bool(getattr(s, "bake_metallic_texture", True))
    wants_emission = bool(getattr(s, "bake_emission_texture", True))
    wants_ao = bool(getattr(s, "bake_ao_texture", True))

    if not (
        wants_normal
        or wants_base_color
        or wants_roughness
        or wants_metallic
        or wants_emission
        or wants_ao
    ):
        raise ValueError("Enable at least one bake/map output.")

    if not low.data.uv_layers:
        raise ValueError("Low poly needs a UV map")

    warnings = configure_cycles_devices(s)
    settings, strategy, max_dim = prepare_settings_and_strategy(high, low, s)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    base_name = low.name[:40]

    bake_material = build_bake_material(base_name, settings)
    low.data.materials.clear()
    low.data.materials.append(bake_material)

    texture_paths = {}

    if wants_normal:
        log("Baking normal detail...")
        texture_paths["NORMAL"] = process_map(
            high,
            low,
            bake_material,
            "NORMAL",
            "NORMAL",
            base_name + "_Normal",
            settings.normal_map_resolution,
            strategy,
            max_dim,
            settings,
            out_dir,
            base_name + "_normal.png"
        )

    if wants_base_color:
        log("Baking base color...")

        base_color_post = None
        if settings.brighter:
            def base_color_post(arr):
                _boost_base_color_array(arr, 1.0 / 1.2)

        metallic_restore = suppress_metallic(high)
        try:
            texture_paths["BASE_COLOR"] = process_map(
                high,
                low,
                bake_material,
                "DIFFUSE",
                "BASE_COLOR",
                base_name + "_BaseColor",
                settings.base_color_resolution,
                strategy,
                max_dim,
                settings,
                out_dir,
                base_name + "_basecolor.png",
                array_post_process=base_color_post
            )
        finally:
            restore_metallic(metallic_restore)

    if wants_roughness:
        log("Baking roughness...")
        texture_paths["ROUGHNESS"] = process_map(
            high,
            low,
            bake_material,
            "ROUGHNESS",
            "ROUGHNESS",
            base_name + "_Roughness",
            getattr(settings, "roughness_resolution", settings.base_color_resolution),
            strategy,
            max_dim,
            settings,
            out_dir,
            base_name + "_roughness.png"
        )

    if wants_metallic:
        log("Baking metallic (via emission swap)...")
        swap_records = swap_metallic_to_emission(high)
        try:
            texture_paths["METALLIC"] = process_map(
                high,
                low,
                bake_material,
                "EMIT",
                "METALLIC",
                base_name + "_Metallic",
                getattr(settings, "metallic_resolution", settings.base_color_resolution),
                strategy,
                max_dim,
                settings,
                out_dir,
                base_name + "_metallic.png"
            )
        finally:
            restore_emission_swap(swap_records)

    if wants_emission:
        log("Baking emission...")
        texture_paths["EMISSION"] = process_map(
            high,
            low,
            bake_material,
            "EMIT",
            "EMISSION",
            base_name + "_Emission",
            getattr(settings, "emission_resolution", settings.base_color_resolution),
            strategy,
            max_dim,
            settings,
            out_dir,
            base_name + "_emission.png"
        )

    if wants_ao:
        log("Baking AO...")
        texture_paths["AO"] = process_map(
            high,
            low,
            bake_material,
            "AO",
            "AO",
            base_name + "_AO",
            getattr(settings, "ao_resolution", settings.base_color_resolution),
            strategy,
            max_dim,
            settings,
            out_dir,
            base_name + "_ao.png",
            samples=max(32, settings.bake_sample_count)
        )

    # Free the high-poly mesh before building/exporting the final low-poly GLB.
    try:
        bpy.data.objects.remove(high, do_unlink=True)
    except Exception:
        pass

    purge_orphan_datablocks()

    export_material = build_export_material(base_name, texture_paths, settings)
    low.data.materials.clear()
    low.data.materials.append(export_material)

    try:
        bpy.data.materials.remove(bake_material)
    except Exception:
        pass

    purge_orphan_datablocks()

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

    result = {"warnings": warnings}

    if "NORMAL" in texture_paths:
        result["normal_png"] = texture_paths["NORMAL"]
    if "BASE_COLOR" in texture_paths:
        result["base_color_png"] = texture_paths["BASE_COLOR"]
    if "ROUGHNESS" in texture_paths:
        result["roughness_png"] = texture_paths["ROUGHNESS"]
    if "METALLIC" in texture_paths:
        result["metallic_png"] = texture_paths["METALLIC"]
    if "EMISSION" in texture_paths:
        result["emission_png"] = texture_paths["EMISSION"]
    if "AO" in texture_paths:
        result["ao_png"] = texture_paths["AO"]

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
                "bake_metallic_texture": ("BOOLEAN", {"default": True}),
                "metallic_resolution": ("INT", {"default": 8192, "min": 16, "max": 32768}),
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
                "material_metallic": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0}),
                "material_specular_level": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
            },
            "optional": {
                "load_output_images": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Load baked PNGs into IMAGE outputs. Disable to reduce ComfyUI memory if you only need the GLB/files."
                }),
            },
        }

    RETURN_TYPES = (
        "FILE_3D_GLB",
        "IMAGE",
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
        "metallic",
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
        load_output_images = bool(kw.pop("load_output_images", True))

        import shutil

        base_out_dir = os.path.join(folder_paths.get_output_directory(), "LODTailorBakeForger", "latest")

        if os.path.isdir(base_out_dir):
            shutil.rmtree(base_out_dir, ignore_errors=True)

        os.makedirs(base_out_dir, exist_ok=True)
        out_dir = base_out_dir

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

        cmd = [
            blender_bin,
            "--factory-startup",
            "-noaudio",
            "-b",
            "-P",
            script_path,
            "--",
            config_path,
        ]

        log_path = os.path.join(out_dir, "blender_process.log")

        try:
            with open(log_path, "wb") as log_file:
                process = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    cwd=out_dir,
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
            tail = ""
            if os.path.exists(log_path):
                try:
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                        tail = fh.read()[-2000:]
                except Exception:
                    tail = ""

            raise RuntimeError("Blender bake failed: " + str(result.get("error", tail)))

        file_3d = _BakedFile3D(result["glb"], "glb")
        info = json.dumps(result, indent=2)

        if load_output_images:
            base_tensor = _load_image_tensor(result.get("base_color_png", ""))
            normal_tensor = _load_image_tensor(result.get("normal_png", ""))
            roughness_tensor = _load_image_tensor(result.get("roughness_png", ""))
            metallic_tensor = _load_image_tensor(result.get("metallic_png", ""))
            emission_tensor = _load_image_tensor(result.get("emission_png", ""))
            ao_tensor = _load_image_tensor(result.get("ao_png", ""))
        else:
            base_tensor = _empty_image_tensor()
            normal_tensor = _empty_image_tensor()
            roughness_tensor = _empty_image_tensor()
            metallic_tensor = _empty_image_tensor()
            emission_tensor = _empty_image_tensor()
            ao_tensor = _empty_image_tensor()

        return (
            file_3d,
            base_tensor,
            normal_tensor,
            roughness_tensor,
            metallic_tensor,
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
