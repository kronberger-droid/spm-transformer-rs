# Active TODOs

Last updated: 2025-12-03

## Phase 1: Baseline ViT ✅ COMPLETE!

### Completed ✅
- [x] Fix #1: Class weights implementation
- [x] Fix #2: Per-scanline normalization
- [x] Fix #3: Learning rate + warmup scheduler
- [x] Fix #4: Number of classes (inferred from data)
- [x] Fix #5: Increase layers to 6
- [x] Test training run with all fixes
- [x] Documentation restructure (.claude/ folder)
- [x] Create separate build.nu script
- [x] Remove unused training.rs file

**Result**: 66.5% validation accuracy (vs 36% baseline) - SUCCESS! 🎉

---

## Phase 2: Address Overfitting ✅ COMPLETE!

**Original Goal**: 70-73% validation accuracy
**Result**: ViT ceiling at ~67% - cannot reach 85% target

### Completed ✅
- [x] Add AdamW optimizer with weight decay
- [x] Add environment variable support for all hyperparameters
- [x] Implement early stopping (MetricEarlyStoppingStrategy)
- [x] Create hyperparameter sweep script (slurm/sweep.nu)
- [x] Run 8-config hyperparameter sweep
  - Weight decay: [0.0, 0.01, 0.05, 0.1]
  - Dropout: [0.1, 0.15, 0.2]

**Result**: Best config (no weight decay, dropout=0.1) achieved 66.74% validation - SUCCESS for ViT optimization!

**Key Finding**: ViT architecture has hit ceiling at ~67%. Need sequential transformer to reach 85% target.

---

## Phase 3: Sequential Transformer Architecture (CURRENT PRIORITY) 🔥

**Problem**: ViT ceiling at 67%, need 85% to match Gordon et al. 2020
**Goal**: Implement sequential transformer with causal masking for sequential scanline data
**Status**: Planning & Design

**See `ARCHITECTURE.md` for detailed design**

### Priority 1: Architecture Design & Planning

- [ ] Review ARCHITECTURE.md and update with latest insights
- [ ] Design CNN tokenizer architecture
  - Input: (batch, 128, pixels_per_line) per scanline
  - Output: (batch, feature_dim) embedding per scanline
  - Decide: kernel sizes, number of layers, pooling strategy
- [ ] Design causal attention modification
  - Research: Burn's TransformerEncoder masking support
  - Plan: How to add causal mask (position i attends only to j ≤ i)
- [ ] Design positional encoding strategy
  - Decide: Sinusoidal vs learned vs relative
  - Plan: Integration with sequential model

### Priority 2: Implementation (In Order)

#### Step 1: CNN Tokenizer (Foundation)
- [ ] Create `src/tokenizer.rs` module
- [ ] Implement 1D CNN for per-scanline feature extraction
  - 3-5 conv layers
  - Kernel sizes: 3, 5, 7
  - Output: fixed-size embedding per scanline
- [ ] Test tokenizer independently (input shape → output shape)
- **Effort**: 2-3 hours

#### Step 2: Causal Attention Mask
- [ ] Research Burn's attention mask API
- [ ] Implement causal masking in transformer
  - Triangular mask: position i only sees j ≤ i
  - Integration with TransformerEncoder
- [ ] Test mask application (verify attention patterns)
- **Effort**: 2-4 hours

#### Step 3: Sequential Model Integration
- [ ] Create new model: `SequentialScanLineEncoder`
- [ ] Integrate CNN tokenizer + causal transformer
- [ ] Add positional encodings
- [ ] Wire up to training loop
- **Effort**: 1-2 hours

#### Step 4: Testing & Validation
- [ ] Test forward pass with dummy data
- [ ] Run small training test (10 epochs)
- [ ] Compare to ViT baseline (should see different learning pattern)
- **Effort**: 1 hour

### Priority 3: Experimentation

#### Hyperparameter Tuning
- [ ] Window size ablation (W = 10, 20, 30, 40, 128)
- [ ] CNN depth ablation (3 vs 5 layers)
- [ ] Feature dimension experiments (128, 256, 512)
- [ ] Positional encoding comparison (sinusoidal vs learned)

#### Analysis
- [ ] Attention pattern visualization
- [ ] Compare to ViT baseline
- [ ] Ablation: with/without causal masking
- [ ] Compare to LRCN baseline (if available)

### Target Metrics
- **Minimum**: 75% validation accuracy (beat ViT by 9 points)
- **Goal**: 85% validation accuracy (match Gordon et al. 2020)
- **Stretch**: 90%+ validation accuracy

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
