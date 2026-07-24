"""
run_zone_blend.py — CLI entry point for the Phase 2 zone-aware blending pipeline.

Loads MoMask models once, then runs the ZoneBlendPipeline for each job in a
JSON jobs file (same format as edit_t2m_batch.py).

Usage
-----
    python scripts/run_zone_blend.py \\
        --jobs_file sweep_jobs.json \\
        --alpha 0.5 \\
        --zone_mode standard \\
        --gpu_id 0

Jobs file format (same schema as edit_t2m_batch.py):
    [
      {
        "ext":           "walk_001/walk_to_dance_q01",
        "text_prompt":   "[walk:0.91] A person walks.",
        "source_motion": "outputs/walk_001/mediapipe_out.npz",
        "repeat_times":  1
      },
      ...
    ]

The source_motion .npz must contain either:
  - 'motion'  : (T, 263) HumanML3D motion vector  (preferred)
  - 'joints'  : (T, 22, 3) raw joint positions    (fallback)
"""

import argparse
import json
import os
import time
from os.path import join as pjoin

import numpy as np
import torch

from gen_t2m import load_res_model, load_trans_model, load_vq_model
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor

from semantic_spectrum.pipeline import ZoneBlendPipeline


def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 2 zone-blend pipeline: feature-space blending of '
                    'original and MoMask-generated motion.'
    )
    p.add_argument('--jobs_file',   required=True,
                   help='Path to JSON jobs file.')
    p.add_argument('--alpha',       type=float, default=0.5,
                   help='Blend strength in [0, 1]. Default: 0.5')
    p.add_argument('--zone_mode',   default='standard',
                   choices=['standard', 'side_specific'],
                   help='Zone configuration mode. Default: standard')
    p.add_argument('--gpu_id',      type=int, default=0)
    p.add_argument('--name',        default='MaskTransformer')
    p.add_argument('--res_name',    default='ResTransformer')
    p.add_argument('--dataset_name', default='t2m')
    p.add_argument('--checkpoints_dir', default='./checkpoints')
    p.add_argument('--time_steps', type=int,   default=18)
    p.add_argument('--cond_scale', type=float, default=4.0)
    p.add_argument('--temperature',type=float, default=1.0)
    p.add_argument('--topkr',      type=float, default=0.9)
    p.add_argument('--skip_ik',    action='store_true')
    p.add_argument('--out_dir',    default='./blend_outputs',
                   help='Root output directory. Default: ./blend_outputs')
    return p.parse_args()


def load_source(path: str) -> tuple[np.ndarray | None, np.ndarray]:
    """
    Load source motion from .npz.

    Returns
    -------
    motion_vec : (T, 263) or None if not present
    joints     : (T, 22, 3)
    """
    data = np.load(path, allow_pickle=True)
    motion_vec = None
    if 'motion' in data:
        motion_vec = data['motion'].astype(np.float32)
    if 'joints' in data:
        joints = data['joints'].astype(np.float32)
    elif motion_vec is not None:
        # derive joints from motion vec
        joints = recover_from_ric(
            torch.from_numpy(motion_vec).float(), 22
        ).numpy()
    else:
        raise ValueError(f"Source file {path} must contain 'motion' or 'joints'.")
    return motion_vec, joints


def main():
    args = parse_args()
    fixseed(10107)

    with open(args.jobs_file) as f:
        jobs = json.load(f)
    if not jobs:
        print("No jobs to run.")
        return

    device    = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    dim_pose  = 251 if args.dataset_name == 'kit' else 263

    # ---- Load models ----
    root_dir       = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt      = get_opt(model_opt_path, device=device)

    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name,
                        model_opt.vq_name, 'opt.txt')
    vq_opt           = get_opt(vq_opt_path, device=device)
    vq_opt.dim_pose  = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens     = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim       = vq_opt.code_dim

    res_opt_path = pjoin(args.checkpoints_dir, args.dataset_name,
                         args.res_name, 'opt.txt')

    class _Opt:
        pass
    opt = _Opt()
    opt.device          = device
    opt.gpu_id          = args.gpu_id
    opt.name            = args.name
    opt.res_name        = args.res_name
    opt.dataset_name    = args.dataset_name
    opt.checkpoints_dir = args.checkpoints_dir
    opt.time_steps      = args.time_steps
    opt.cond_scale      = args.cond_scale
    opt.temperature     = args.temperature
    opt.topkr           = args.topkr
    opt.gumbel_sample   = False
    opt.force_mask      = False

    res_opt   = get_opt(res_opt_path, device=device)
    res_model = load_res_model(res_opt, vq_opt, opt)
    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    for m in (vq_model, t2m_transformer, res_model):
        m.eval()
        m.to(device)

    mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name,
                         model_opt.vq_name, 'meta', 'mean.npy'))
    std  = np.load(pjoin(args.checkpoints_dir, args.dataset_name,
                         model_opt.vq_name, 'meta', 'std.npy'))

    converter = Joint2BVHConvertor()

    # ---- Build pipeline ----
    pipeline = ZoneBlendPipeline(
        vq_model=vq_model,
        mask_transformer=t2m_transformer,
        res_model=res_model,
        vq_opt=vq_opt,
        mean=mean,
        std=std,
        alpha=args.alpha,
        zone_mode=args.zone_mode,
        device=device,
        time_steps=args.time_steps,
        cond_scale=args.cond_scale,
        temperature=args.temperature,
        topkr=args.topkr,
    )

    print(f"\nModels loaded. Running {len(jobs)} job(s) "
          f"[alpha={args.alpha}, zone_mode={args.zone_mode}] ...\n")
    t0 = time.time()

    for i, job in enumerate(jobs):
        torch.cuda.empty_cache()
        ext         = job['ext']
        text_prompt = job['text_prompt']
        src_path    = job['source_motion']

        print(f"\n[{i+1}/{len(jobs)}] {ext}")
        print(f"  prompt : {text_prompt}")

        result_dir    = pjoin(args.out_dir, ext)
        joints_dir    = pjoin(result_dir, 'joints')
        animation_dir = pjoin(result_dir, 'animations')
        os.makedirs(joints_dir,    exist_ok=True)
        os.makedirs(animation_dir, exist_ok=True)

        t_job = time.time()
        motion_vec, orig_joints = load_source(src_path)

        if motion_vec is not None:
            output_joints = pipeline.run_from_motion_vec(
                motion_vec, orig_joints, text_prompt)
        else:
            output_joints = pipeline.run(orig_joints, text_prompt)

        T = len(output_joints)

        # Save joints
        npy_path = pjoin(joints_dir, 'blend_output.npy')
        np.save(npy_path, output_joints)

        # BVH + mp4 (non-IK)
        bvh_path = pjoin(animation_dir, f'blend_len{T}.bvh')
        _, smooth_joints = converter.convert(output_joints, filename=bvh_path,
                                             iterations=100, foot_ik=False)
        mp4_path = pjoin(animation_dir, f'blend_len{T}.mp4')
        plot_3d_motion(mp4_path, t2m_kinematic_chain, smooth_joints,
                       title='', fps=20)

        if not args.skip_ik:
            bvh_ik = pjoin(animation_dir, f'blend_len{T}_ik.bvh')
            _, ik_joints = converter.convert(output_joints, filename=bvh_ik,
                                             iterations=100, foot_ik=True)
            plot_3d_motion(pjoin(animation_dir, f'blend_len{T}_ik.mp4'),
                           t2m_kinematic_chain, ik_joints, title='', fps=20)

        print(f"  done in {time.time() - t_job:.1f}s → {result_dir}")

    total = time.time() - t0
    print(f"\nAll {len(jobs)} jobs done in {total:.1f}s ({total/len(jobs):.1f}s/job)")


if __name__ == '__main__':
    main()
