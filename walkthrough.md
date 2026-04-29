# Walkthrough: Rescue Implementation

I have fully implemented the rescue plan to address the evaluation failures. The codebase has been updated to fix the mode collapse in the decoder and the distribution shift in the diffusion model.

## 1. Waveform Decoder Rescued
- **Capacity Increased**: In `src/models_diffusion.py`, I increased the `hidden_dim` of the 1D-CNN from 128 to **256**. This gives the network the mathematical capacity needed to learn the complex waves of all 6 activities without blurring them together.
- **Smoothness Fixed**: In `src/train_waveform_decoder.py`, I drastically reduced the default `smoothness_weight` from `0.5` to **0.005**. The network will no longer be terrified of outputting sharp, dynamic peaks (like those seen in jogging), preventing the flatline behavior.

## 2. Diffusion Training Upgraded
- **Cosine Schedule**: I replaced the standard linear DDPM noise schedule with a **Cosine Schedule** across `train_diffusion.py`, `evaluate_downstream.py`, and `evaluate_classifier.py`. Cosine schedules prevent the latent tokens from being destroyed too quickly in the early forward steps, which is critical for complex continuous signals.
- **Batch Size**: Increased default `batch_size` to 256 for smoother gradient updates.

## 3. Evaluation Suite Enhanced
I heavily updated `evals/eval_master_suite.py`:
- **Comparative Waveform Plots**: Added a function that extracts a random physical signal and a synthetic signal for every single class, and plots them **overlaid** on the X, Y, and Z axes. They are saved to `results/waveforms/` and logged to WandB.
- **Class Imbalance Repair Experiment**: Ported the rigorous imbalance test from the old scripts. It starves Class 0 ('Downstairs') to just 5%, trains a baseline, then uses the diffusion model to generate synthetic replacements, trains a repaired model, and prints comparative F1 scores and Confusion Matrices.
- **CFG Tuning**: Added a `--cfg_weight` argument so you can sweep values directly from the command line without editing the code.

---

## What to Run Next

Because we fundamentally changed the noise schedule (to Cosine) and the decoder architecture (to 256 dims), you **must retrain both models** before evaluating.

Run these three commands in order (your 4-hour blocks are fine, they are decoupled):

**1. Train Diffusion Model:**
```bash
python src/train_diffusion.py --data features/biopm_tokens.npz --epochs 300 --wandb
```

**2. Train Waveform Decoder:**
```bash
python src/train_waveform_decoder.py --data features/biopm_tokens.npz --epochs 150 --wandb
```

**3. Run the Master Evaluation (with Sweep):**
Let's test the default `cfg_weight=1.5` first. If the PCA plot still shows red dots on the edge, drop the weight to `0.5` or `0.0` (unconditional).
```bash
python evals/eval_master_suite.py --data features/biopm_tokens.npz --diff_ckpt checkpoints/diffusion/token_diff_ema.pt --dec_ckpt checkpoints/diffusion/waveform_decoder.pt --out_dir results/ --cfg_weight 1.5 --wandb
```
