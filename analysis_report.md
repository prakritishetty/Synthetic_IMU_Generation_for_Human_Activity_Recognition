# Synthetic IMU Pipeline: Evaluation Analysis Report

This report breaks down the recent evaluation results of the generative pipeline. While the architectural fixes successfully prevented the strict mode-collapse observed in previous iterations, the model is currently suffering from severe underfitting and distribution mismatch.

## 1. Distributional Shifts (PCA Manifold Analysis)
**Observation:**
The PCA scatter plot shows the Real tokens (blue) forming distinct, dense clusters corresponding to the physical activity manifolds. However, the Synthetic tokens (red) are uniformly scattered across the entire latent space like a random cloud, failing to align with any specific real cluster.

**Reasoning:**
The TokenTransformer Diffusion model has failed to learn the conditional distributions. It is essentially failing to completely denoise the initial Gaussian noise $x_T$, resulting in outputs that are still highly noisy. Because the synthetic tokens are mathematically random noise rather than structured BioPM tokens, they scatter uniformly in the PCA space.

## 2. Comparative Waveform Visualizations
**Observation:**
When observing the overlaid waveforms across the X, Y, and Z axes:
- **Sitting (Class 2)**: The Real waveform is a perfectly flat line (expected for a stationary activity). The Synthetic waveform oscillates wildly with high amplitude.
- **Jogging / Walking**: The Real waveforms show sharp, high-frequency periodic impacts (footsteps). The Synthetic waveforms are wandering and heavily smoothed out, failing to capture the sharp impact peaks.

**Reasoning:**
The `WaveformDecoder` is an unconditional translator: its only job is to map a 64-d token into a physical wave. Because the Diffusion model is feeding it *random noise tokens* (as proven by the PCA plot), the decoder is translating that noise into random, meaningless physical waves. The model hallucinates movement for the "Sitting" class because the input tokens for "Sitting" were incorrectly generated as high-variance noise by the failing diffusion model. 

## 3. Physical Sanity (Amplitude & Smoothness)
**Observation:**
- **Sitting Smoothness**: Real is `0.0106` (very still), but Synthetic is `0.1707` (very jittery).
- **Amplitude Histograms**: The synthetic distributions fail to match the real distributions, often clustering around zero and missing the true bi-modal peaks or heavy tails of the physical data.

**Reasoning:**
This further corroborates the waveform visual analysis. The generated synthetic data does not possess the physical statistics of human movement because the underlying token generation is currently failing. 

## 4. Downstream Utility & Class Imbalance Repair
### The Experimental Setup
To prove the utility of synthetic data, we designed the **Class Imbalance Repair** experiment:
1. **Baseline**: We deliberately starve the dataset by removing 95% of all "Downstairs" (Class 0) data, simulating a severe real-world data collection shortage. We train a Random Forest classifier on this starved dataset.
2. **Repair**: We use our Diffusion Model to generate exactly 944 purely synthetic "Downstairs" samples to replace the missing data. We add this synthetic data to the starved training set and train a new "Repaired" Random Forest classifier.

### The Results
**Observation:**
The Baseline model achieved an F1 score of `0.7010`, completely failing to classify any "Downstairs" samples (0 True Positives in the confusion matrix). 
The Repaired model achieved an F1 score of `0.7032`. The confusion matrix shows it *still* scored 0 True Positives for "Downstairs". 

**Reasoning:**
Because the diffusion model generated random noise instead of distinct "Downstairs" tokens, injecting this synthetic data into the Random Forest training set provided no useful discriminative boundaries. The classifier simply ignored the synthetic noise, resulting in an identical F1 score.

---

## Next Steps for Improvement

The pipeline architecture is now completely sound (the decoupled Waveform Decoder, the Cosine Schedule, and CFG are mathematically correct). The failure lies entirely in the **convergence of the TokenTransformer Diffusion Model**. 

To fix this, we must:
1. **Increase Training Epochs**: Cosine noise schedules generally require significantly longer training times than linear schedules. We should increase diffusion epochs from 300 to 1000.
2. **Tune Learning Rate**: The learning rate (`2e-4`) might be too high or too low for the new batch size (`256`) and the Cosine schedule. We must inspect the `diffusion_loss` curve in WandB to ensure it is actually trending toward zero.
3. **Verify CFG Weight**: A CFG weight of $w=1.5$ might be too aggressive if the unconditional model hasn't converged. We should test $w=0.0$ (pure conditional) to isolate whether the issue is guidance-related or base-model-related.
