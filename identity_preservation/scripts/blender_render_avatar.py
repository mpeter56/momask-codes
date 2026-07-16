"""
blender_render_avatar.py — Headless Blender pipeline.

Loads character.fbx + a BVH, retargets the BVH motion onto the character's
armature via Copy-Rotation constraints, bakes the animation, renders to MP4.

Two modes:
  --introspect             Print character bones and exit (no render)
  --bvh X.bvh --out X.mp4  Full pipeline (retarget + bake + render)

Invocation (macOS Blender):
  /Applications/Blender.app/Contents/MacOS/Blender \
      -b --python .../blender_render_avatar.py -- \
      --fbx ~/Downloads/character.fbx --introspect

  /Applications/Blender.app/Contents/MacOS/Blender \
      -b --python .../blender_render_avatar.py -- \
      --fbx ~/Downloads/character.fbx \
      --bvh .../blended_d0.40.bvh \
      --out .../avatar_d0.40.mp4
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import bpy


# ------------------------------------------------------------------
# Bone-name mappings. Ted's template.bvh uses plain SMPL names:
#   Hips, LeftUpLeg, LeftLeg, LeftFoot, LeftToe,
#   RightUpLeg, RightLeg, RightFoot, RightToe,
#   Spine, Spine1, Spine2, Neck, Head,
#   LeftShoulder, LeftArm, LeftForeArm, LeftHand,
#   RightShoulder, RightArm, RightForeArm, RightHand
# ------------------------------------------------------------------
SMPL_BONES = [
    "Hips",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
    "Spine", "Spine1", "Spine2", "Neck", "Head",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
]

# Common alternate namings — character.fbx might use any of these
ALIASES = {
    # Mixamo prefix
    "mixamorig:{name}":            lambda n: f"mixamorig:{n}",
    "mixamorig_{name}":            lambda n: f"mixamorig_{n}",
    # Blender Rigify DEF- prefix (partial)
    "DEF-{name_lower}":            lambda n: f"DEF-{n.lower()}",
    # ReadyPlayerMe / plain
    "{name}":                      lambda n: n,
    # spine variants
}

# Manual SMPL→other special-case mappings (used as last resort)
FUZZY_HINTS = {
    "Hips":         ["Hips", "Pelvis", "Root", "Bip01"],
    "Spine":        ["Spine", "Spine1", "spine_01"],
    "Spine1":       ["Spine1", "Spine2", "spine_02"],
    "Spine2":       ["Spine2", "Chest", "spine_03"],
    "Neck":         ["Neck", "neck_01"],
    "Head":         ["Head", "head"],
    "LeftShoulder": ["LeftShoulder", "L_Shoulder", "shoulder_l", "clavicle_l"],
    "LeftArm":      ["LeftArm", "L_Arm", "upperarm_l", "arm_l"],
    "LeftForeArm":  ["LeftForeArm", "L_ForeArm", "lowerarm_l", "forearm_l"],
    "LeftHand":     ["LeftHand", "L_Hand", "hand_l"],
    "LeftUpLeg":    ["LeftUpLeg", "L_UpLeg", "thigh_l", "upperleg_l"],
    "LeftLeg":      ["LeftLeg", "L_Leg", "calf_l", "lowerleg_l"],
    "LeftFoot":     ["LeftFoot", "L_Foot", "foot_l"],
    "LeftToe":      ["LeftToe", "LeftToeBase", "mixamorig:LeftToeBase", "L_Toe", "ball_l", "toe_l"],
    "RightShoulder":["RightShoulder", "R_Shoulder", "shoulder_r", "clavicle_r"],
    "RightArm":     ["RightArm", "R_Arm", "upperarm_r", "arm_r"],
    "RightForeArm": ["RightForeArm", "R_ForeArm", "lowerarm_r", "forearm_r"],
    "RightHand":    ["RightHand", "R_Hand", "hand_r"],
    "RightUpLeg":   ["RightUpLeg", "R_UpLeg", "thigh_r", "upperleg_r"],
    "RightLeg":     ["RightLeg", "R_Leg", "calf_r", "lowerleg_r"],
    "RightFoot":    ["RightFoot", "R_Foot", "foot_r"],
    "RightToe":     ["RightToe", "RightToeBase", "mixamorig:RightToeBase", "R_Toe", "ball_r", "toe_r"],
}


# ------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--fbx", required=True, help="path to character.fbx")
    ap.add_argument("--bvh", help="path to BVH (omit with --introspect)")
    ap.add_argument("--out", help="output MP4 path")
    ap.add_argument("--introspect", action="store_true", help="dump character bones and exit")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--scale", type=float, default=1.0, help="BVH scale factor (try 100 for cm-unit rigs)")
    ap.add_argument("--view", choices=["front", "back", "left", "right"], default="front",
                    help="camera position relative to character (default: front)")
    ap.add_argument("--zoom", type=float, default=3.5,
                    help="camera distance as multiple of character height (larger = zoomed out)")
    return ap.parse_args(argv)


def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def find_armatures():
    return [o for o in bpy.data.objects if o.type == "ARMATURE"]


def dump_bones(armature):
    print(f"\n=== BONES in armature '{armature.name}' ({len(armature.data.bones)} total) ===")
    for bone in armature.data.bones:
        parent = bone.parent.name if bone.parent else "ROOT"
        print(f"  {bone.name}   (parent: {parent})")


def detect_mapping(char_bones):
    """Return dict {smpl_bone_name: char_bone_name}, best-effort."""
    char_bone_set = set(char_bones)
    mapping = {}

    # Try each alias template
    for smpl_name in SMPL_BONES:
        best = None
        # 1. exact match
        if smpl_name in char_bone_set:
            best = smpl_name
        # 2. mixamorig prefix
        elif f"mixamorig:{smpl_name}" in char_bone_set:
            best = f"mixamorig:{smpl_name}"
        elif f"mixamorig_{smpl_name}" in char_bone_set:
            best = f"mixamorig_{smpl_name}"
        # 3. fuzzy candidate list
        else:
            for candidate in FUZZY_HINTS.get(smpl_name, []):
                if candidate in char_bone_set:
                    best = candidate; break
                # case-insensitive
                for cb in char_bones:
                    if cb.lower() == candidate.lower():
                        best = cb; break
                if best: break

        if best:
            mapping[smpl_name] = best
    return mapping


# ------------------------------------------------------------------
# Pipeline stages
# ------------------------------------------------------------------
def import_fbx(path):
    bpy.ops.import_scene.fbx(filepath=path)
    arms = find_armatures()
    if not arms:
        raise RuntimeError(f"No armature in {path}")
    return arms[-1]


def import_bvh(path, scale=1.0, fps=20):
    before = set(find_armatures())
    bpy.ops.import_anim.bvh(
        filepath=path,
        target='ARMATURE',
        global_scale=scale,
        frame_start=1,
        use_fps_scale=False,
        update_scene_fps=False,
        update_scene_duration=True,
    )
    after = set(find_armatures())
    new_arms = after - before
    if not new_arms:
        raise RuntimeError(f"BVH import failed to add armature: {path}")
    return next(iter(new_arms))


def apply_retarget(char_armature, bvh_armature, mapping):
    bpy.context.view_layer.objects.active = char_armature
    bpy.ops.object.mode_set(mode="POSE")

    for smpl_bone, char_bone in mapping.items():
        pbone = char_armature.pose.bones.get(char_bone)
        if pbone is None:
            print(f"  skip {smpl_bone} -> {char_bone} (not in pose bones)")
            continue
        c = pbone.constraints.new(type="COPY_ROTATION")
        c.target = bvh_armature
        c.subtarget = smpl_bone
        # Rest-pose mismatch fix: apply BVH rotation ON TOP of the character's
        # rest pose instead of replacing it. Without this, Y Bot's arms snap
        # to SMPL's T-pose because Copy Rotation just copies raw orientation.
        if hasattr(c, "mix_mode"):
            c.mix_mode = "BEFORE"           # Blender 3.0+
        elif hasattr(c, "use_offset"):
            c.use_offset = True             # Blender 2.x fallback
        c.target_space = "LOCAL"
        c.owner_space = "LOCAL"

    # Skip COPY_LOCATION on Hips. BVH's Hips world position (SMPL rest ~0.93m)
    # would override Y Bot's rest hip height and detach feet from the ground,
    # producing the "floating character" look. Character animates in place;
    # feet stay planted via Y Bot's own rest position + foot-IK in the BVH.

    bpy.ops.object.mode_set(mode="OBJECT")


def bake_and_remove_source(char_armature, bvh_armature):
    scene = bpy.context.scene
    end_frame = scene.frame_end
    bpy.context.view_layer.objects.active = char_armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.nla.bake(
        frame_start=1,
        frame_end=end_frame,
        only_selected=True,
        visual_keying=True,
        clear_constraints=True,
        clear_parents=False,
        use_current_action=True,
        bake_types={"POSE"},
    )
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.data.objects.remove(bvh_armature, do_unlink=True)


def setup_scene_render(char_armature, out_path, fps, width, height,
                       view="front", zoom=3.5):
    """Configure scene + camera + lighting + PNG image-sequence output.

    Returns the temp dir where PNG frames will be written (main() converts to MP4
    via system ffmpeg after render, because this Blender build lacks FFmpeg
    support in bpy.render.image_settings).
    """
    scene = bpy.context.scene

    # Auto-frame based on character bounding box, so head + feet both fit.
    all_meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    import mathutils
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

    # Mixamo characters imported via bpy.ops.import_scene.fbx face +Y after
    # Blender's default axis conversion (Y-up FBX -> Z-up Blender flips fwd).
    # Empirically for Y Bot / X Bot: "front" means looking from +Y toward -Y.
    dist = char_h * zoom
    if view == "front":
        cam_pos = (center.x,           center.y + dist, center.z + char_h * 0.05)
    elif view == "back":
        cam_pos = (center.x,           center.y - dist, center.z + char_h * 0.05)
    elif view == "left":
        cam_pos = (center.x - dist,    center.y,        center.z + char_h * 0.05)
    else:  # right
        cam_pos = (center.x + dist,    center.y,        center.z + char_h * 0.05)

    print(f"[camera] view={view}  bbox_h={char_h:.2f}  center={tuple(center)}  cam={cam_pos}")

    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.object
    scene.camera = cam
    # Aim at pelvis-ish (mid-height) so the full body fits vertically
    bpy.ops.object.empty_add(location=(center.x, center.y, center.z))
    aim = bpy.context.object
    aim.name = "cam_target"
    track = cam.constraints.new("TRACK_TO")
    track.target = aim
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    # Lighting
    bpy.ops.object.light_add(type="SUN", location=(2, -2, 4))
    bpy.context.object.data.energy = 3.0
    bpy.ops.object.light_add(type="AREA", location=(-2, -2, 3))
    bpy.context.object.data.energy = 200

    # World background — soft gray so it reads like Ted's stick-figure style
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    try:
        world.use_nodes = True   # (deprecated in Blender 6, but OK for now)
        bg = world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.9, 0.9, 0.92, 1.0)
    except Exception:
        pass

    # Render engine — pick whatever this Blender build supports
    engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "CYCLES"):
        if candidate in engines:
            scene.render.engine = candidate
            break

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.fps = fps

    # Output: PNG image sequence into a temp subdir next to the target MP4.
    # main() will combine with system ffmpeg after render finishes.
    out_dir = os.path.dirname(os.path.abspath(out_path))
    stem = os.path.splitext(os.path.basename(out_path))[0]
    frame_dir = os.path.join(out_dir, f"_frames_{stem}")
    if os.path.isdir(frame_dir):
        shutil.rmtree(frame_dir)
    os.makedirs(frame_dir, exist_ok=True)

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = os.path.join(frame_dir, "frame_")   # frame_0001.png, ...
    return frame_dir


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    clear_scene()

    char_armature = import_fbx(args.fbx)
    print(f"[fbx] loaded '{args.fbx}', armature: {char_armature.name}")
    dump_bones(char_armature)

    if args.introspect:
        print("\n[introspect] done — no render performed.")
        return 0

    if not args.bvh or not args.out:
        raise SystemExit("--bvh and --out are required unless --introspect")

    bvh_armature = import_bvh(args.bvh, scale=args.scale, fps=args.fps)
    print(f"[bvh] loaded '{args.bvh}', armature: {bvh_armature.name}")

    # Pin frame range to the BVH's actual animation length (not Blender's default 250)
    n_frames = 250
    if bvh_armature.animation_data and bvh_armature.animation_data.action:
        n_frames = int(bvh_armature.animation_data.action.frame_range[1])
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = n_frames
    print(f"[bvh] animation range: 1..{n_frames}")

    char_bones = [b.name for b in char_armature.data.bones]
    mapping = detect_mapping(char_bones)
    print(f"\n[mapping] resolved {len(mapping)}/{len(SMPL_BONES)} SMPL bones:")
    for smpl, char in mapping.items():
        print(f"    {smpl:15s} -> {char}")
    missing = [b for b in SMPL_BONES if b not in mapping]
    if missing:
        print(f"[mapping] MISSING: {missing}")
    if len(mapping) < 10:
        raise SystemExit(f"[FATAL] only {len(mapping)}/{len(SMPL_BONES)} bones mapped — inspect bone names and add to FUZZY_HINTS")

    apply_retarget(char_armature, bvh_armature, mapping)
    bake_and_remove_source(char_armature, bvh_armature)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame_dir = setup_scene_render(char_armature, args.out, args.fps, args.width, args.height,
                                   view=args.view, zoom=args.zoom)

    print(f"\n[render] engine={bpy.context.scene.render.engine}  frames -> {frame_dir}")
    bpy.ops.render.render(animation=True)
    n_frames = len([f for f in os.listdir(frame_dir) if f.endswith(".png")])
    print(f"[render] wrote {n_frames} PNG frames")

    # ---- combine PNG frames -> MP4 via system ffmpeg -----------------------
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[warn] ffmpeg not on PATH — leaving PNG frames at {frame_dir}")
        print(f"[done] frames saved: {frame_dir}")
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
    print(f"[ffmpeg] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[fatal] ffmpeg failed: {e}")
        return 3

    # cleanup frames — comment out if you want to keep them for slides
    shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"[done] saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
