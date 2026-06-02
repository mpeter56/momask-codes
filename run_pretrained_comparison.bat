@echo off
REM Run all 40 prompts with pretrained (untuned) models for comparison
REM Output goes to editing/pretrained_* folders

set MASK=t2m_nlayer8_nhead6_ld384_ff1024_cdp0.1_rvq6ns
set RES=tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw

REM --- Group 1: Tags + Text ---
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.09 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.09][walk:0.91] A person walks forward." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.11 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.11][walk:0.91] A person walks forward." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.28 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.28][walk:0.91] A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.39 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.39][walk:0.91] A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.40 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.40][walk:0.91] A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.51 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.51][walk:0.91] A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.63 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.63][walk:0.91] A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.74 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.74][walk:0.91] A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_0.80 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.80][walk:0.91] A person dances expressively." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_text_1.00 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:1.00][walk:0.91] A person dances expressively." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1

REM --- Group 2: Text Only ---
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.09 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person walks forward." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.11 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person walks forward." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.28 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.39 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.40 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.51 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.63 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.74 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances with a walking base." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_0.80 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances expressively." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_text_only_1.00 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "A person dances expressively." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1

REM --- Group 3: Tags Only ---
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.09 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.09][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.11 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.11][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.28 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.28][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.39 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.39][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.40 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.40][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.51 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.51][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.63 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.63][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.74 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.74][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_0.80 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.80][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_tags_only_1.00 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:1.00][walk:0.91]" --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1

REM --- Group 4: Spectrum Edit (10% steps) ---
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d10 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.10][walk:0.90] A person moves with 10% dance and 90% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d30 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.30][walk:0.70] A person moves with 30% dance and 70% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d40 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.40][walk:0.60] A person moves with 40% dance and 60% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d50 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.50][walk:0.50] A person moves with 50% dance and 50% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d60 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.60][walk:0.40] A person moves with 60% dance and 40% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d70 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.70][walk:0.30] A person moves with 70% dance and 30% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d80 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:0.80][walk:0.20] A person moves with 80% dance and 20% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1
python edit_t2m.py --gpu_id 0 --ext pretrained_spectrum_d100 --name %MASK% --dataset_name t2m --res_name %RES% --text_prompt "[dance:1.00][walk:0.00] A person moves with 100% dance and 0% walk." --source_motion momask_input.npz --mask_edit_section 0.0,1.0 --use_res_model --repeat_times 1

echo All pretrained comparison runs complete.
