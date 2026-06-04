"""Batch version of edit_t2m.py — loads models once, runs all jobs sequentially.

Accepts a JSON jobs file so the caller (verb_sweep.py) can submit all prompts
in a single process rather than spawning one subprocess per prompt.

Jobs file format (list of dicts):
    [
      {
        "ext":           "walk_001/walk_to_dance_s1_fixed_q01of10",
        "text_prompt":   "[walk:0.91][dance:0.09] A person walks.",
        "source_motion": "outputs/walk_001/mediapipe_out.npz",
        "repeat_times":  1
      },
      ...
    ]

Usage:
    python edit_t2m_batch.py \\
        --jobs_file sweep_jobs.json \\
        --gpu_id 0 \\
        --name MaskTransformer \\
        --res_name ResTransformer \\
        --dataset_name t2m \\
        --mask_edit_section 0.0,1.0
"""

import argparse
import json
import os
import time
from os.path import join as pjoin

import numpy as np
import torch

from gen_t2m import load_res_model, load_trans_model, load_vq_model
from options.eval_option import EvalT2MOptions
from utils.fixseed import fixseed
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.paramUtil import t2m_kinematic_chain
from utils.plot_script import plot_3d_motion
from visualization.joints2bvh import Joint2BVHConvertor


def parse_batch_args():
    parser = argparse.ArgumentParser(
        description='Batch edit_t2m: load models once, run all jobs from a JSON file.'
    )
    parser.add_argument('--jobs_file',  required=True,
                        help='Path to JSON file listing all inference jobs.')
    parser.add_argument('--gpu_id',     type=int, default=0)
    parser.add_argument('--name',       default='MaskTransformer')
    parser.add_argument('--res_name',   default='ResTransformer')
    parser.add_argument('--dataset_name', default='t2m')
    parser.add_argument('--checkpoints_dir', default='./checkpoints')
    parser.add_argument('--mask_edit_section', default='0.0,1.0',
                        help='Comma-separated start,end fraction for edit mask.')
    parser.add_argument('--time_steps', type=int, default=18)
    parser.add_argument('--cond_scale', type=float, default=4.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--topkr',      type=float, default=0.9)
    parser.add_argument('--gumbel_sample', action='store_true')
    parser.add_argument('--force_mask', action='store_true')
    return parser.parse_args()


def run_job(job, models, opt, mean, std, converter, edit_section_str):
    ext          = job['ext']
    text_prompt  = job['text_prompt']
    source_motion_path = job['source_motion']
    repeat_times = job.get('repeat_times', 1)

    vq_model, t2m_transformer, res_model, vq_opt = models

    result_dir    = pjoin('./editing', ext)
    joints_dir    = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir,    exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    # Check already done
    import glob
    existing = glob.glob(pjoin(animation_dir, '**', '*.mp4'), recursive=True)
    if existing:
        print(f"  Skipping {ext} (already has {len(existing)} mp4s)")
        return

    # Load source motion
    motion_data = np.load(source_motion_path, allow_pickle=True)
    motion = motion_data['motion'] if isinstance(motion_data, np.lib.npyio.NpzFile) else motion_data
    print(f"  Loaded motion shape: {motion.shape}")

    m_length = len(motion)
    motion = motion.astype(np.float32)
    motion = (motion - mean) / std

    max_motion_length = 196
    if max_motion_length > m_length:
        motion = np.concatenate(
            [motion, np.zeros((max_motion_length - m_length, motion.shape[1]))], axis=0
        )
    motion_tensor = torch.from_numpy(motion)[None].to(opt.device)

    captions   = [text_prompt]
    token_lens = torch.div(
        torch.LongTensor([m_length]), 4, rounding_mode='floor'
    ).to(opt.device)
    m_length_t = token_lens * 4

    # Edit mask
    _start_str, _end_str = edit_section_str.split(',')
    edit_start, edit_end = float(_start_str), float(_end_str)

    with torch.no_grad():
        tokens, _ = vq_model.encode(motion_tensor)

    edit_mask = torch.zeros_like(tokens[..., 0])
    seq_len   = tokens.shape[1]
    s_idx = int(edit_start * seq_len)
    e_idx = int(edit_end   * seq_len)
    edit_mask[:, s_idx:e_idx] = 1
    edit_mask = edit_mask.bool()

    print_caption = f"{text_prompt} [{edit_start}-{edit_end}]"
    kinematic_chain = t2m_kinematic_chain

    def inv_transform(data):
        return data * std + mean

    for r in range(repeat_times):
        print(f"  -->Repeat {r}")
        with torch.no_grad():
            mids = t2m_transformer.edit(
                captions, tokens[..., 0].clone(), token_lens,
                timesteps=opt.time_steps,
                cond_scale=opt.cond_scale,
                temperature=opt.temperature,
                topk_filter_thres=opt.topkr,
                gsample=opt.gumbel_sample,
                force_mask=opt.force_mask,
                edit_mask=edit_mask.clone(),
            )
            mids = res_model.generate(mids, captions, token_lens, temperature=1, cond_scale=2)
            pred_motions  = vq_model.forward_decoder(mids)
            pred_motions  = pred_motions.detach().cpu().numpy()
            source_np     = motion_tensor.detach().cpu().numpy()
            data          = inv_transform(pred_motions)
            source_data_t = inv_transform(source_np)

        for k, (caption, joint_data, src) in enumerate(zip(captions, data, source_data_t)):
            print(f"  ---->Sample {k}: {caption} {m_length_t[k]}")
            anim_path  = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir,    str(k))
            os.makedirs(anim_path,  exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            ml = int(m_length_t[k])
            joint_data   = joint_data[:ml]
            joint        = recover_from_ric(torch.from_numpy(joint_data).float(), 22).numpy()
            src          = src[:ml]
            source_joint = recover_from_ric(torch.from_numpy(src).float(), 22).numpy()

            bvh = pjoin(anim_path, f"sample{k}_repeat{r}_len{ml}_ik.bvh")
            _, ik_joint = converter.convert(joint, filename=bvh, iterations=100)
            bvh2 = pjoin(anim_path, f"sample{k}_repeat{r}_len{ml}.bvh")
            _, joint = converter.convert(joint, filename=bvh2, iterations=100, foot_ik=False)

            plot_3d_motion(pjoin(anim_path, f"sample{k}_repeat{r}_len{ml}_ik.mp4"),
                           kinematic_chain, ik_joint, title=print_caption, fps=20)
            plot_3d_motion(pjoin(anim_path, f"sample{k}_repeat{r}_len{ml}.mp4"),
                           kinematic_chain, joint, title=print_caption, fps=20)
            plot_3d_motion(pjoin(anim_path, f"sample{k}_source_len{ml}.mp4"),
                           kinematic_chain, source_joint, title='None', fps=20)
            np.save(pjoin(joint_path, f"sample{k}_repeat{r}_len{ml}.npy"),    joint)
            np.save(pjoin(joint_path, f"sample{k}_repeat{r}_len{ml}_ik.npy"), ik_joint)


def main():
    args = parse_batch_args()
    fixseed(10107)

    with open(args.jobs_file) as f:
        jobs = json.load(f)

    if not jobs:
        print("No jobs to run.")
        return

    device = torch.device("cpu" if args.gpu_id == -1 else f"cuda:{args.gpu_id}")
    dim_pose = 251 if args.dataset_name == 'kit' else 263

    # ---- Load models ONCE ----
    root_dir       = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    model_opt_path = pjoin(root_dir, 'opt.txt')

    class _Opt:
        pass
    opt = _Opt()
    opt.device         = device
    opt.gpu_id         = args.gpu_id
    opt.name           = args.name
    opt.res_name       = args.res_name
    opt.dataset_name   = args.dataset_name
    opt.checkpoints_dir = args.checkpoints_dir
    opt.time_steps     = args.time_steps
    opt.cond_scale     = args.cond_scale
    opt.temperature    = args.temperature
    opt.topkr          = args.topkr
    opt.gumbel_sample  = args.gumbel_sample
    opt.force_mask     = args.force_mask

    model_opt = get_opt(model_opt_path, device=device)

    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=device)
    vq_opt.dim_pose = dim_pose
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens     = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim       = vq_opt.code_dim

    res_opt_path = pjoin(args.checkpoints_dir, args.dataset_name, args.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=device)
    res_model = load_res_model(res_opt, vq_opt, opt)

    t2m_transformer = load_trans_model(model_opt, opt, 'latest.tar')

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    res_model.to(device)
    t2m_transformer.to(device)
    vq_model.to(device)

    mean = np.load(pjoin(args.checkpoints_dir, args.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std  = np.load(pjoin(args.checkpoints_dir, args.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    converter = Joint2BVHConvertor()
    models    = (vq_model, t2m_transformer, res_model, vq_opt)

    print(f"\nModels loaded. Running {len(jobs)} job(s) ...\n")
    t0 = time.time()

    for i, job in enumerate(jobs):
        print(f"\n[{i+1}/{len(jobs)}] {job['ext']}")
        print(f"  prompt: {job['text_prompt']}")
        t_job = time.time()
        run_job(job, models, opt, mean, std, converter, args.mask_edit_section)
        print(f"  job time: {time.time() - t_job:.1f}s")

    total = time.time() - t0
    print(f"\nAll {len(jobs)} jobs done in {total:.1f}s  ({total/len(jobs):.1f}s/job)")


if __name__ == '__main__':
    main()
