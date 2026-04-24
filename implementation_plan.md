# Bio-PM-Guided Synthetic IMU Generation

This project builds an end-to-end pipeline to extract BioPM token-level representations from IMU data, train a lightweight generative model to produce synthetic token sequences, and test those sequences on downstream tasks. 

**GitHub Integration:** We will initialize the workspace as a git repository and link it to your remote: `https://github.com/prakritishetty/Synthetic_IMU_Generation_for_Human_Activity_Recognition`.

## Oomph Factor: Continuous Gen-AI Diffusion Model
We will implement an unconditional and **Class-Conditioned Continuous Diffusion Model**. 
To clarify your question: **Yes, this does true synthetic generation!** Just like Stable Diffusion generates new images from noise, our model will learn the statistical distribution of real Bio-PM tokens. At inference time, it will take in **pure random Gaussian noise** (and a target Activity label like 'Walking') and iteratively denoise it to "dream up" **100% novel, hallucinated token sequences**. These sequences will then be decoded into physical raw IMU waveforms. It is fundamentally a synthetic generator, not an autoencoder reconstruction tool.

## Proposed Changes

> [!NOTE] 
> To differentiate our new scripts from the provided toolkit, all your original starter scripts will be prepended with `starter_` (e.g., `starter_preprocess_data.py`). Our new scripts will have clean domain names.

### Data Setup and Preprocessing
#### [NEW] `scripts/download_wisdm.py`
- Fetches and unpacks the standard WISDM v1.1 dataset locally into `raw_data/WISDM/`.

#### [NEW] `src/preprocess_wisdm.py`
- A cloned and modified version of the starter preprocessing codebase.
- Parses the WISDM comma-separated `.txt` format and remaps the string labels.
- Resamples the 20 Hz WISDM dataset up to the 30 Hz expected by BioPM.

### Token Extraction
#### [NEW] `src/extract_tokens.py`
- Bypasses the heavy pooling (`masked_mean_std`) found in the starter feature extractor.
- Runs inference on the pre-trained `BioPMModel` and saves the continuous `(N, L, 64)` matrix of tokens, alongside sequence validities and pad masks for diffusion training.

### Generative Diffusion & Decoders
#### [NEW] `src/models_diffusion.py`
- **Class-Conditioned Token Diffusion Model**: A tiny U-Net or Transformer designed to reverse the sequence noise process based on chosen activity classes.
- **IMU Movement Decoder**: An MLP that takes generated 64-d tokens and maps them back to the physical 32-sample raw temporal patches.
- *Note:* We use the pre-trained Bio-PM strictly as a frozen Feature Extractor; these generative modules operate on top of its outputs.

#### [NEW] `src/train_diffusion.py`
- Script to train the Token Diffusion Model and the IMU decoder.
- Features **extensive logging** via `wandb` (Weights & Biases) and `tqdm` progress bars.
- Designed to be exceptionally lightweight, running very fast on local CPU for immediate testing, with PyTorch device scaling for subsequent T4 GPU training.

### Deliverables & Evaluation
#### [NEW] `src/evaluate_downstream.py`
- **Synthetic v. Real Analysis**: Generates t-SNE / UMAP plots mapping our pure synthetically sampled tokens vs. the real Bio-PM distribution.
- **Downstream Utility Experiment**: Instead of Subject ID (which is incompatible with label-blind synthetics unless explicitly conditioned!), we will tackle **Class Imbalance**. The original project asked for class balancing, so we will artificially restrict training samples for a rare class (e.g., 'Stairs'). We will then generate synthetic 'Stairs' IMU data and demonstrate exactly how blending our synthetic data repairs the `Activity Classifier` F1 Macro score.

## Open Questions

> [!WARNING]
> Since we're incorporating Weights & Biases (`wandb`) for extensive logging, it will ask for an API key the first time you run the training script. Please make sure you have a W&B account ready or let me know if you would like to run it in offline mode initially.

## Verification Plan

### Automated Tests
- `download_wisdm.py` -> `preprocess_wisdm.py` -> `extract_tokens.py` to ensure valid `(N, max_len, 64)` token sequences.
- Local mock run of `train_diffusion.py` on your CPU to ensure the diffusion loss converges without crashing and `tqdm`/`wandb` operate smoothly.

### Manual Verification
- Review the remote GitHub commits to ensure the codebase pushes successfully.
- Review the generated UMAP artifacts and the final Imbalanced F1 Macro Score comparison (Real-Imbalanced vs. Real-Imbalanced + Synthetic Data) for Activity Classification.
