# Active TODOs

Last updated: 2025-12-02

## Errors:
- [ ] Fix: rust cant create checkpoint dir. Might not be mounted to container?

## Phase 1: Baseline ViT (Current)

### Completed ✅
- [x] Fix #1: Class weights implementation
- [x] Fix #2: Per-scanline normalization
- [x] Fix #3: Learning rate + warmup scheduler
- [x] Fix #4: Number of classes (inferred from data)
- [x] Documentation restructure (.claude/ folder)

### In Progress 🔄
- [ ] Fix #5: Increase layers to 6
- [ ] Test training run with all fixes

---

## Phase 2: Incremental Improvements (If ViT showms promise)

### Training Improvements
- [ ] Add weight decay to AdamW (e.g., 0.05)
- [ ] Add gradient clipping (max_norm=1.0)
- [ ] Implement early stopping
- [ ] Add ReduceLROnPlateau after warmup

### Data Improvements
- [ ] Add online data augmentation
  - Horizontal flip
  - Gaussian noise
  - Line dropout
- [ ] Verify dataset quality
- [ ] Check for data leakage between splits

### Model Improvements
- [ ] Experiment with different d_model (128, 256, 512)
- [ ] Try different num_heads (4, 8, 16)
- [ ] Test different dropout rates (0.05, 0.1, 0.2)

---

## Phase 3: Sequential Transformer Architecture (If ViT plateaus)

**See `ARCHITECTURE.md` for detailed plan**

### Core Changes
- [ ] Implement 1D CNN tokenizer
  - Per-scanline feature extraction
  - 3-5 conv layers
  - Kernel sizes: 3, 5, 7
- [ ] Add causal attention mask
  - Modify TransformerEncoder to support masking
  - Ensure position i only attends to j ≤ i
- [ ] Replace learned positional embeddings with sinusoidal
  - Fixed, non-learned encoding
  - Better for variable-length sequences
- [ ] Test progressive prediction
  - Per-token classification
  - Early stopping when confident

### Experimentation
- [ ] Window size ablation (W = 10, 20, 30, 40, 128)
- [ ] CNN depth ablation (3 vs 5 layers)
- [ ] Attention pattern visualization
- [ ] Compare to LRCN baseline

---

## Infrastructure TODOs

### Code Quality
- [x] Remove unused `TrainingConfig` struct
- [x] Add `#[allow(dead_code)]` for `is_empty()` or remove it
- [ ] Add more logging/debugging output
- [ ] Add unit tests for data loading
- [ ] Add unit tests for model forward pass

### Documentation
- [ ] Add inline code comments for complex sections

### Tooling
- [ ] Add training visualization script (plot loss/accuracy)
- [ ] Add checkpoint analysis script (compare runs)
- [ ] Add attention visualization tool (for sequential transformer)
- [ ] Add inference script for single images

---

## Blockers & Questions

### Current Blockers
None - all critical path items are unblocked.

### Open Questions
1. **Should we add weight decay now or later?**
   - Recommendation: Add after seeing baseline results

2. **What batch size for cluster training?**
   - Current: 32 (matches Python)
   - Consider: 64 or 128 if GPU memory allows

4. **Should we implement ReduceLROnPlateau now?**
   - Current: Linear warmup only
   - Recommendation: Add if baseline shows promise

---

## Nice-to-Have (Low Priority)

- [ ] Add mixed precision training (FP16)
- [ ] Add distributed training support
- [ ] Add model export to ONNX
- [ ] Add real-time inference demo
- [ ] Add web UI for model inspection
