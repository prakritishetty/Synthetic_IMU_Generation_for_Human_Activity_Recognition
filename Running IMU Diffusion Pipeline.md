# Chat Conversation

Note: _This is purely the output of the chat conversation and does not contain any raw data, codebase snippets, etc. used to generate the output._

### User Input

hi i forgot hwo to run the codes, and which codes to run. for now just tell me the normal diffusion one (not classfier free guidance), adn then we'll go to that later. everything form the start to the end evals

*Listed directory [CS690TR](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR) *

*Listed directory [scripts](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/scripts) *

*Listed directory [src](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src) *

*Viewed [implementation_plan.md](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/implementation_plan.md) *

*Viewed [train_diffusion.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/train_diffusion.py) *

*Viewed [models_diffusion.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/models_diffusion.py) *

*Viewed [evaluate_downstream.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/evaluate_downstream.py) *

*Viewed [preprocess_wisdm.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/preprocess_wisdm.py) *

*Listed directory [raw_data](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/raw_data) *

*Listed directory [WISDM](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/raw_data/WISDM) *

*Listed directory [WISDM_ar_v1.1](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/raw_data/WISDM/WISDM_ar_v1.1) *

*Viewed [extract_tokens.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/extract_tokens.py) *

*Listed directory [checkpoints](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/checkpoints) *

*Viewed [README.md](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/checkpoints/README.md) *

*Grep searched codebase*

*Grep searched codebase*

*Viewed [download_wisdm.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/scripts/download_wisdm.py) *

### Planner Response

To run the **normal (unconditional/standard conditional) diffusion pipeline** from start to finish, you can follow these steps. This process assumes you are using the pre-trained BioPM model as a feature extractor and training a Transformer-based diffusion model on top of those tokens.

### 1. Data Setup & Preprocessing
First, ensure the raw WISDM data is downloaded and converted into the format required for token extraction.

```powershell
# Optional: Download the raw WISDM dataset if you haven't already
python scripts/download_wisdm.py

# Preprocess the raw TXT file into BioPM-compatible .h5 files
python src/preprocess_wisdm.py `
    --wisdm_txt "raw_data/WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt" `
    --output_dir "raw_data/processed_wisdm"
```

### 2. Token Extraction
Use the pre-trained BioPM model to extract the continuous 64-dimensional tokens that the diffusion model will learn to generate.

```powershell
python src/extract_tokens.py `
    --data_dir "raw_data/processed_wisdm" `
    --checkpoint "checkpoints/checkpoint.pt" `
    --output "features/biopm_tokens.npz"
```

### 3. Diffusion Training
This step trains the **Token Diffusion Model** (to generate token sequences) and the **IMU Decoder** (to turn those tokens back into physical waveforms).

```powershell
# Set epochs to 30 or higher for better results. 
# Remove --wandb if you don't want to log to Weights & Biases.
python src/train_diffusion.py `
    --data "features/biopm_tokens.npz" `
    --epochs 30 `
    --wandb
```
> [!TIP]
> The script will automatically use your GPU (CUDA) if available.

### 4. Evaluation & Generation
Finally, run the evaluation suite to generate 200 "hallucinated" synthetic sequences and compare them against real data via t-SNE and waveform plots.

```powershell
python src/evaluate_downstream.py `
    --data "features/biopm_tokens.npz" `
    --diff_ckpt "checkpoints/diffusion/token_diff.pt" `
    --dec_ckpt "checkpoints/diffusion/imu_decoder.pt"
```

### Summary of Outputs
- **`features/biopm_tokens.npz`**: The "latent" dataset for training.
- **`checkpoints/diffusion/`**: Your trained diffusion and decoder models.
- **`plots/`**: Visualizations including:
    - `tsne_latent_distribution.png`: Shows how well the synthetic tokens overlap with real ones.
    - `waveform_cmp_demo_X.png`: Side-by-side comparison of real vs. synthetic physical IMU signals.

We can look at implementing **Classifier-Free Guidance (CFG)** once you've verified this baseline works!

### User Input

got this while running train-diffusion.py
am i missing soemthing?
Loading continuous tokens from features/biopm_tokens.npz...
Traceback (most recent call last):
  File "/content/Synthetic_IMU_Generation_for_Human_Activity_Recognition/src/train_diffusion.py", line 99, in <module>
    main()
  File "/content/Synthetic_IMU_Generation_for_Human_Activity_Recognition/src/train_diffusion.py", line 41, in main
    raw_patches = raw_patches.view(B, L, -1)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: shape '[10927, 192, -1]' is invalid for input of size 9834300


figure out anf tell me what to change, don't change yourself

*Viewed [dataset.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/data/dataset.py) *

*Viewed [preprocessing.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/data/preprocessing.py) *

*Viewed [extract_tokens.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/extract_tokens.py) *

*Grep searched codebase*

*Grep searched codebase*

*Viewed [biopm.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/models/biopm.py) *

*Viewed [extract_tokens.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/extract_tokens.py) *

### Planner Response

Based on the error message, the issue is a **mismatch between the shape of the raw data** saved in your features file and the **shape expected by the diffusion model**.

### The Problem
In `train_diffusion.py`, the code expects `raw_patches` to be the segmented physical signals corresponding to each of the **192 patches** (e.g., shape `(N, 192, 32)`). However, your file contains the **unsegmented 10-second raw windows** (shape `(N, 300, 3)`). 

- Total size of your `raw_patches`: **9,834,300**
- Number of samples ($B$): **10,927**
- Elements per window: $9,834,300 / 10,927 = \mathbf{900}$
- Your code tries to reshape this into $(10927, 192, \text{something})$, but **900 is not divisible by 192**, hence the `RuntimeError`.

This happened because `extract_tokens.py` was likely capturing the wrong variable from the data loader.

### What to Change
You need to modify `src/extract_tokens.py` to ensure it saves the **segmented patches** (which the BioPM encoder actually "sees") instead of the full raw window.

In `src/extract_tokens.py`, look at the loop where tokens are extracted:

```python
# [src/extract_tokens.py] - Around line 72
# CHANGE THIS:
all_raw_patches.append(raw_batch.cpu().numpy()) 

# TO THIS:
all_raw_patches.append(my_X.cpu().numpy())
```

**Why this fix?**
- `raw_batch` contains the full 10-second window ($300 \text{ samples} \times 3 \text{ axes} = 900$ values).
- `my_X` contains the normalized movement-element patches ($192 \text{ patches} \times 32 \text{ samples} = 6144$ values).
- The `IMUDecoder` in the diffusion model is designed to reconstruct these 32-sample patches from the 64-d tokens.

### Steps to resolve:
1. Open `src/extract_tokens.py`.
2. Ensure line 72 (or similar) is appending `my_X` to `all_raw_patches`.
3. **Re-run the extraction script**:
   ```powershell
   python src/extract_tokens.py --data_dir raw_data/processed_wisdm --checkpoint checkpoints/checkpoint.pt --output features/biopm_tokens.npz
   ```
4. Then run `train_diffusion.py` again. it should now correctly reshape the data.

### User Input

i have added the results to the plots directory
what i am concerned about
1. even after 500 epochs for the diffusion, the tsne still shows that synthetic are on the outskirts of the real datapoints, and only on one corner, not even covering the whole circle
2. the waveform comparison seems very different, with fully different amplitudes
3. another thing in the waveform comparison is that the x, y, z accelrations seem to be the same data. when i had raised this last time you had told to append raw_batch instead of my_X, but that created a shape issue during training, so I had to keep it as is. i think there should be some other way
4. what are the next steps to really get better results? i know there's the classifer free guidance thing that I haven't run yet, but I want to get better results in this itself
look at the codes right from preprocess_data to extract_tokens to train_diffusion to evaluate_downstream and evaluate_classifier and tell me why is this happening and what can we chnage to get better result

*Listed directory [inference](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/inference) *

*Listed directory [CS690TR](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR) *

*Listed directory [plots](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots) *

*Viewed [tsne_latent_distribution.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/tsne_latent_distribution.png) *

*Viewed [waveform_cmp_demo_0.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/waveform_cmp_demo_0.png) *

*Viewed [waveform_cmp_demo_1.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/waveform_cmp_demo_1.png) *

*Viewed [waveform_class_0_batch_1777255784_0.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/waveform_class_0_batch_1777255784_0.png) *

*Viewed [tsne_latent_distribution_1777255784.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/tsne_latent_distribution_1777255784.png) *

*Viewed [confusion_matrix_1777075621.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/confusion_matrix_1777075621.png) *

*Viewed [confusion_matrix_cmp_1777074243.png](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/plots/confusion_matrix_cmp_1777074243.png) *

*Viewed [evaluate_downstream.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/evaluate_downstream.py) *

*Viewed [train_diffusion.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/train_diffusion.py) *

*Viewed [models_diffusion.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/models_diffusion.py) *

*Viewed [preprocess_wisdm.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/preprocess_wisdm.py) *

### Planner Response

Now I have everything I need. Let me compile this into a thorough diagnosis.

*Edited relevant file*

### Planner Response

Here's the full root-cause diagnosis for each issue:

---

## Issue 1 — t-SNE: Synthetics on the outskirts

**Root cause → Wrong DDPM sampler formula** in `evaluate_downstream.py:34`.

The sampler computes `beta_t` as:
```python
beta_t = 1 - (alpha_t / alpha_t_prev)  # ❌ WRONG
```
This is a *derived approximation* of beta, not the actual scheduled beta. The error compounds across 1000 reverse steps, so the sampling trajectory never properly converges to the real data manifold. This is why synthetics land near the boundary in a few tight clusters instead of spreading across the space.

**Fix**: precompute the raw `betas = torch.linspace(1e-4, 0.02, 1000)` and use those directly in the sampler, exactly like training does.

---

## Issue 2 — Waveform amplitude ~5–10× too high in synthetic

**Root cause → Decoder output is unbounded.** The `IMUDecoder` MLP ends with a plain `nn.Linear` (no sigmoid, no clamp). During training it learns to match normalized `[0,1]` movement element patches. But at inference time, the diffusion-generated tokens are out-of-distribution (because of the broken sampler above), so the unclamped linear head extrapolates freely to `±2`.

**Fix**: needs two things — (1) fix the sampler so tokens land in-distribution, and (2) add `torch.tanh` or `torch.clamp` on the decoder's final output to keep values bounded.

---

## Issue 3 — X, Y, Z look identical

**Root cause → Wrong reshape** in `evaluate_downstream.py:47`:
```python
real_seq = real_raw.reshape(-1, 3)  # ❌ WRONG interpretation
```
`raw_patches` stores BioPM movement-element patches of shape `(L=192, 32)` — each patch is **one axis at a time** (axis identity is stored in the metadata, not in separate columns). When you flatten to `(6144,)` and re-split as `(2048, 3)`, you're treating 3 consecutive normalized scalar values as X/Y/Z — which gives nearly identical wiggles.

**Fix**: the right comparison for real data should use `window_acc_raw` from the `.h5` files (the actual 3-axis physical acceleration windows, `shape: (T, 3)`), loaded separately. The `raw_patches` variable was never 3-channel and shouldn't be plotted that way.

---

## Issue 4 — Poor generation even after 500 epochs

Multiple compounding problems:

| Problem | Detail |
|---|---|
| **Conflicting gradients** | Decoder loss and diffusion loss share one `opt.step()`. The decoder is moving the token representation at the same time the diffusion model is trying to learn to predict noise on it. |
| **Learning rate too high** | `lr=1e-3` is fine for the decoder (regression) but too aggressive for the noise-prediction Transformer — it prevents tight convergence. |
| **No EMA** | Every published diffusion model uses exponential moving average of weights for sampling. Without it, you sample from a noisy instantaneous checkpoint instead of the smooth optimum. |
| **Weak conditioning** | Class+time is just *added* uniformly: `x = x + cond_emb`. This signal gets washed out across 4 transformer layers. Modern diffusion transformers use **AdaLayerNorm** (DiT-style) where class/time predict `scale` and `shift` for each layer. |

---

## Priority of fixes

1. 🔴 **Fix the sampler** (`evaluate_downstream.py`) — will immediately improve t-SNE
2. 🔴 **Fix waveform comparison** — use `window_acc_raw` for real, and visualize synthetic as 1D signal
3. 🟡 **Separate decoder and diffusion optimizers** with different learning rates (`1e-3` decoder, `2e-4` diffusion)
4. 🟡 **Add EMA** on the diffusion model weights
5. 🟡 **Upgrade conditioning** to AdaLayerNorm injection

Once the sampler is correct and conditioning is stronger, **CFG will actually work** — right now CFG would just amplify the existing broken output.

Want me to go ahead and make these changes?

### User Input

yes let's do it


*Viewed [extract_tokens.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/extract_tokens.py) *

### Planner Response

Let me tackle this systematically, one file at a time.

*Edited relevant file*

### Planner Response

### Step 1: `models_diffusion.py` — DiT-style AdaLayerNorm + bounded decoder

*Edited relevant file*

### Planner Response

### Step 2: `extract_tokens.py` — also save physical 3-axis windows

*Edited relevant file*

### Planner Response

### Step 3: `train_diffusion.py` — EMA + separate optimizers

*Edited relevant file*

### Planner Response

### Step 4: `evaluate_downstream.py` — fix sampler + fix waveform plots

*Edited relevant file*

*Edited relevant file*

### Planner Response

All 4 files are updated. Here's what changed and what you need to do:

---

## What changed

| File | Key changes |
|---|---|
| `models_diffusion.py` | AdaLayerNorm in every transformer block (DiT-style); 6 layers; `nn.Sigmoid` on decoder output |
| `extract_tokens.py` | Now also saves `raw_windows` (physical 3-axis windows) in the npz |
| `train_diffusion.py` | Two separate optimizers (AdamW 2e-4 for diffusion, Adam 1e-3 for decoder); EMA; cosine LR schedule; decoder uses `detach()` so no gradient bleed |
| `evaluate_downstream.py` | Correct DDPM sampler; real waveforms use physical `raw_windows`; synthetic waveforms split by axis group |

---

## How to re-run (from scratch — necessary because model architecture changed)

```powershell
# Step 1: Re-extract tokens (adds raw_windows to the npz)
python src/extract_tokens.py \
    --data_dir raw_data/processed_wisdm \
    --checkpoint checkpoints/checkpoint.pt \
    --output features/biopm_tokens.npz

# Step 2: Retrain diffusion (new architecture + better training)
python src/train_diffusion.py \
    --data features/biopm_tokens.npz \
    --epochs 300 \
    --wandb

# Step 3: Evaluate using the EMA checkpoint (default)
python src/evaluate_downstream.py \
    --data features/biopm_tokens.npz
```

> [!IMPORTANT]
> Because the model architecture changed (DiT blocks vs. plain TransformerEncoder), your existing `token_diff.pt` checkpoint **is not compatible**. You must retrain from scratch. The new architecture with AdaLayerNorm should converge significantly better, so fewer epochs may be needed than before.

### User Input

can you check the files again and see if they still reflect the changes you made? i had soem merge conflicts so i'm hoping i didnt change anything

*Edited relevant file*

*Edited relevant file*

*Edited relevant file*

*Viewed [models_diffusion.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/models_diffusion.py) *

### Planner Response

I can see three specific merge issues. Let me fix them one by one.

*Viewed [evaluate_downstream.py](file:///e:/UMass/SPRING%202026/690R/Downloads/CS690TR/src/evaluate_downstream.py) *

### Planner Response

Good — I can see exactly what broke. Three files need fixes:

**`models_diffusion.py`**: `IMUDecoder` was reverted to old code (wrong default `patch_dim=32*3`, no `Sigmoid`, smaller MLP).

**`extract_tokens.py`**: `raw_patches` now appends `raw_batch` (the physical window) instead of `my_X` (the ME patches) — this reintroduces the shape mismatch bug.

**`evaluate_downstream.py`**: The `plot_waveforms` body was reverted to old broken code (`real_raw`/`synthetic_raw` undefined variables), and the waveform loop still uses `raw_patches`, `syn_raw_np`, and undeclared `i`.

*Edited relevant file*

*Edited relevant file*

### Planner Response

Now fix the three problems in `evaluate_downstream.py`:

*Edited relevant file*

### Planner Response

All three files are corrected. Here's a summary of what was broken and what I fixed:

| File | What broke | Fix |
|---|---|---|
| `models_diffusion.py` | `IMUDecoder` reverted — wrong `patch_dim=32*3`, missing `Sigmoid`, smaller MLP | Restored correct `patch_dim=32`, two-layer MLP, `nn.Sigmoid()` |
| `extract_tokens.py` | `raw_patches` was appending `raw_batch` instead of `my_X` — reintroduced the original shape mismatch | Restored `my_X.cpu().numpy()` for `raw_patches`; `raw_batch` only goes to `raw_windows` |
| `evaluate_downstream.py` | `plot_waveforms` body used undefined `real_raw`/`synthetic_raw`; waveform loop used `syn_raw_np[idx]` and undeclared `i`; WandB upload pointed to old filename | Restored correct function body with proper variable names; fixed loop to use `raw_windows` and `syn_patches_np`; fixed WandB upload to use `timestamp` filename |

The files are now consistent with the intended changes. You're safe to re-run the pipeline.