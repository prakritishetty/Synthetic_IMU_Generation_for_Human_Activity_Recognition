python src/preprocess_wisdm.py --wisdm_txt raw_data/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt --output_dir data/preprocessed

python src/extract_tokens.py --data_dir data/preprocessed --checkpoint checkpoints/checkpoint.pt --output features/biopm_tokens.npz

# Phase 2
python src/train_diffusion.py --data features/biopm_tokens.npz --epochs 700 --wandb

python src/train_waveform_decoder.py --data features/biopm_tokens.npz --epochs 400 --wandb


# Phase 3
python evals/eval_master_suite.py --data features/biopm_tokens.npz --diff_ckpt checkpoints/diffusion/token_diff_ema.pt --dec_ckpt checkpoints/diffusion/waveform_decoder.pt --out_dir results/ --cfg_weight 1.5 --wandb

