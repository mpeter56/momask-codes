"""
rokoko_render_avatar.py — headless retarget via Rokoko Studio Live plugin, then render.

Uses the Rokoko plugin's Python API (bpy.ops.rsl.build_bone_list /
retarget_animation) instead of raw Copy-Rotation constraints, which handles
rest-pose deltas correctly. Then renders a PNG sequence and combines via ffmpeg.

Invocation (from pipeline.sh):
  Blender -b --python rokoko_render_avatar.py -- \
      --fbx Y\\ Bot.fbx --bvh blended_d0.40.bvh --out avatar_d0.40.mp4 \
      --rokoko-zip ~/Downloads/rokoko.zip

If Rokoko is already installed, --rokoko-zip is optional. If not installed and
--rokoko-zip is provided, this script installs + enables it before use.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback

import bpy


ROKOKO_MODULE_CANDIDATES = (
    "rokoko-studio-live-blender",
    "rokoko_studio_live_blender",
    "rokoko",
    "rsl",
)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", required=True)
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rokoko-zip", default="")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--view", choices=["front", "back", "left", "right"], default="front")
    ap.add_argument("--zoom", type=float, default=3.5)
    return ap.parse_args(argv)


# ------------------------------------------------------------------
# Rokoko plugin ensure
# ------------------------------------------------------------------
def rokoko_enabled_module() -> str:
    for a in bpy.context.preferences.addons:
        mod = a.module.lower()
        if "rokoko" in mod or "rsl" == mod:
            return a.module
    return ""


def ensure_rokoko(zip_path: str) -> str:
    """Return the enabled module name; install from zip if needed."""
    mod = rokoko_enabled_module()
    if mod:
        print(f"[rokoko] already enabled: {mod}")
        return mod

    if zip_path and os.path.isfile(zip_path):
        print(f"[rokoko] installing from {zip_path}")
        bpy.ops.preferences.addon_install(overwrite=True, filepath=zip_path)

    # Try enabling common module names
    for cand in ROKOKO_MODULE_CANDIDATES:
        try:
            bpy.ops.preferences.addon_enable(module=cand)
            print(f"[rokoko] enabled: {cand}")
            return cand
        except Exception as e:
            print(f"[rokoko] enable {cand} failed: {e}")

    # Last resort — scan installed addons for anything Rokoko-ish
    import addon_utils
    for m in addon_utils.modules():
        name = getattr(m, "__name__", "")
        if "rokoko" in name.lower() or "rsl" in name.lower():
            try:
                bpy.ops.preferences.addon_enable(module=name)
                print(f"[rokoko] enabled: {name}")
                return name
            except Exception:
                pass

    raise SystemExit(
        "[FATAL] Rokoko plugin not installed and could not be enabled.\n"
        "Install manually via Edit > Preferences > Add-ons > Install… or provide --rokoko-zip"
    )


# ------------------------------------------------------------------
# Scene setup / helpers
# ------------------------------------------------------------------
def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_armatures():
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def import_fbx(path):
    before = set(find_armatures())
    bpy.ops.import_scene.fbx(filepath=path)
    new_arms = set(find_armatures()) - before
    if not new_arms:
        raise RuntimeError(f"No armature in {path}")
    return next(iter(new_arms))


def import_bvh(path, fps=20):
    before = set(find_armatures())
    bpy.ops.import_anim.bvh(
        filepath=path,
        target='ARMATURE',
        global_scale=1.0,
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
    )
    new_arms = set(find_armatures()) - before
    if not new_arms:
        raise RuntimeError(f"BVH import produced no armature: {path}")
    return next(iter(new_arms))


# ------------------------------------------------------------------
# Rokoko retarget
# ------------------------------------------------------------------
def rokoko_retarget(char_armature, bvh_armature):
    scene = bpy.context.scene

    # Set source (BVH) and target (character)
    scene.rsl_retargeting_armature_source = bvh_armature
    scene.rsl_retargeting_armature_target = char_armature

    # Auto-detect bone mapping
    print("[rokoko] building bone list...")
    bpy.ops.rsl.build_bone_list()

    mapped = 0
    total = 0
    for item in scene.rsl_retargeting_bone_list:
        total += 1
        if item.bone_name_target:
            mapped += 1
            print(f"    {item.bone_name_source:25s} -> {item.bone_name_target}")
    print(f"[rokoko] mapping: {mapped}/{total} bones")

    if mapped < 10:
        raise SystemExit(f"[FATAL] Rokoko only mapped {mapped}/{total} bones — inspect naming")

    # Retarget — this bakes the animation onto the target armature
    print("[rokoko] retargeting animation...")
    bpy.ops.rsl.retarget_animation()
    print("[rokoko] done")


# ------------------------------------------------------------------
# Camera / lighting / render (same as raw-copy version)
# ------------------------------------------------------------------
def setup_scene_render(char_armature, out_path, fps, width, height, view, zoom):
    import mathutils
    scene = bpy.context.scene

    all_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if all_meshes:
        min_co = mathutils.Vector((1e9, 1e9, 1e9))
        max_co = mathutils.Vector((-1e9, -1e9, -1e9))
        for m in all_meshes:
            for v in m.bound_box:
                world_v = m.matrix_world @ mathutils.Vector(v)
                for i in range(3):
                    min_co[i] = min(min_co[i], world_v[i])
                    max_co[i] = max(max_co[i], world_v[i])
        center = (min_co + max_co) / 2
        char_h = max_co[2] - min_co[2]
    else:
        center = mathutils.Vector((0.0, 0.0, 1.0))
        char_h = 1.8

    dist = char_h * zoom
    if view == "front":
        cam_pos = (center.x,          center.y + dist, center.z + char_h * 0.05)
    elif view == "back":
        cam_pos = (center.x,          center.y - dist, center.z + char_h * 0.05)
    elif view == "left":
        cam_pos = (center.x - dist,   center.y,        center.z + char_h * 0.05)
    else:
        cam_pos = (center.x + dist,   center.y,        center.z + char_h * 0.05)

    print(f"[camera] view={view} bbox_h={char_h:.2f} center={tuple(center)} cam={cam_pos}")

    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.object
    scene.camera = cam
    bpy.ops.object.empty_add(location=(center.x, center.y, center.z))
    aim = bpy.context.object
    aim.name = "cam_target"
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Lights
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
    bpy.context.object.data.energy = 3.0
    bpy.ops.object.light_add(type="AREA", location=(-2, -2, 3))
    bpy.context.object.data.energy = 200

    # World bg
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.9, 0.9, 0.92, 1.0)
    except Exception:
        pass

    # Pick engine
    engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if candidate in engines:
            scene.render.engine = candidate
            break

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.fps = fps

    out_dir = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0]
    frame_dir = os.path.join(out_dir, f"_frames_{stem}")
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = os.path.join(frame_dir, "frame_")
    return frame_dir


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()

    clear_scene()

    # Enable Rokoko once — read_factory_settings doesn't touch addons
    ensure_rokoko(args.rokoko_zip)

    # Sanity check: are the Rokoko scene props actually registered?
    scene = bpy.context.scene
    for prop in ("rsl_retargeting_armature_source",
                 "rsl_retargeting_armature_target",
                 "rsl_retargeting_bone_list"):
        if not hasattr(scene, prop):
            raise SystemExit(f"[FATAL] Rokoko prop '{prop}' missing on scene — "
                             f"plugin registered but did not install its properties. "
                             f"Likely Blender 5.x incompatibility.")
    print("[rokoko] scene properties verified: rsl_retargeting_armature_source/target/bone_list")

    # Import character + BVH
    char_armature = import_fbx(args.fbx)
    print(f"[fbx] loaded, char armature: {char_armature.name}")
    bvh_armature = import_bvh(args.bvh, fps=args.fps)
    print(f"[bvh] loaded, bvh armature: {bvh_armature.name}")

    # Pin frame range to BVH duration
    n_frames = 250
    if bvh_armature.animation_data and bvh_armature.animation_data.action:
        n_frames = int(bvh_armature.animation_data.action.frame_range[1])
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = n_frames
    print(f"[bvh] frame range: 1..{n_frames}")

    # Retarget via Rokoko
    rokoko_retarget(char_armature, bvh_armature)

    # Remove source BVH armature — animation is baked on target
    bpy.data.objects.remove(bvh_armature, do_unlink=True)

    # Set up camera + lights + render output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame_dir = setup_scene_render(char_armature, args.out, args.fps,
                                   args.width, args.height, args.view, args.zoom)

    print(f"[render] engine={bpy.context.scene.render.engine} frames -> {frame_dir}")
    bpy.ops.render.render(animation=True)
    n_png = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    print(f"[render] wrote {n_png} PNG frames")

    # Combine to MP4
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[warn] ffmpeg not on PATH — leaving PNG frames at {frame_dir}")
        return 0

    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-framerate", str(args.fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "20",
        args.out,
    ]
    print(f"[ffmpeg] combining -> {args.out}")
    subprocess.run(cmd, check=True)
    shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"[done] saved: {args.out}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        print("\n[FATAL] uncaught exception in rokoko_render_avatar:")
        traceback.print_exc()
        rc = 99
    sys.exit(rc or 0)
