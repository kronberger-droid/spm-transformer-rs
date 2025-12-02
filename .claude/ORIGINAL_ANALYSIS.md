# STM Tip State Classification Project

## Goal

Replicate the performance of **Gordon et al. 2020** ("Embedding Human Heuristics in Machine Learning for Probe Microscopy") which achieved:

- **Mean AUROC: 0.960**
- **Balanced Accuracy: 0.847**

Using an **LRCN (Long-Term Recurrent Convolutional Network)** architecture on H:Si(100) STM images.

### Target Architecture (Gordon et al. 2020)

**Best Model: CNN + LSTM Hybrid**
- **Window size**: W=20 scan lines (rolling window over 128-line images)
- **CNN**: VGG-like architecture (32, 64, 128 filters; 3×3 kernels; max pooling)
- **LSTM**: 256 hidden units for temporal context
- **Classes**: 4 tip states
  1. Individual atoms (highest/atomic resolution)
  2. Asymmetries/Dimers (combined category)
  3. Rows (lower resolution)
  4. Generic defects (tip quality issues)

**Key Innovation**: Processes consecutive scan lines in windows, using LSTM to learn temporal evolution (mimics human operator behavior)

---

## Current State

### Implementation 1: Python Vision Transformer (Reference)
**Location**: `~/Programming/python/transformer-line-classification/`

**Architecture**: Vision Transformer (ViT)
- Treats each scanline as a token (128 scanlines → 128 tokens)
- 6 transformer encoder layers
- 8 attention heads, 256 embedding dims, 1024 FFN dims
- 0.1 dropout, pre-normalization

**Performance**:
- **Validation Accuracy: 39.23%** (epoch 39)
- **Validation AUROC: 0.6633**
- Training stopped at epoch 54 (early stopping)

**Training Details**:
- Optimizer: AdamW (lr=1e-4, weight_decay=0.05)
- LR Schedule: 5-epoch warmup + ReduceLROnPlateau
- Class weights: [2.845, 0.569, 1.478, 0.823]
- Per-scanline normalization (zero mean, unit variance)
- Augmentation: horizontal flip, Gaussian noise, line dropout
- Batch size: 32

**Status**: Significantly underperforms target (39% vs 85% balanced accuracy)

---

### Implementation 2: Rust Vision Transformer (Current)
**Location**: `/home/kronberger/Programming/rust/spm-transformer/`

**Architecture**: Vision Transformer (ViT-based, called `ScanLineEncoder`)
- Similar to Python version: scanline-as-token approach
- 4 transformer encoder layers (fewer than Python's 6)
- 8 attention heads, 256 embedding dims
- 0.1 dropout, pre-normalization
- Uses Burn framework (supports CPU/CUDA)

**Performance**:
- **Validation Accuracy: ~36%** (epoch 50)
- **Training Accuracy: ~36%** (epoch 50)
- Loss plateaued around 1.60-1.61

**Current Issues**:

1. **❌ Class weights NOT implemented**
   - Declared but passes `None` to loss function (line 147, 157 in model.rs)
   - Critical for handling severe class imbalance (Class 1 is 43.9% of data)

2. **❌ Learning rate too high with no scheduling**
   - Uses constant LR of 0.001 (10× higher than Python's 0.0001)
   - No warmup, no decay, no adaptive scheduling

3. **❌ No data normalization**
   - Uses raw float32 pixel values
   - Python uses per-scanline normalization (mean=0, std=1)

4. **❌ Fewer transformer layers**
   - 4 layers vs Python's 6 layers

5. **⚠️ Number of classes unclear**
   - Model hardcoded to 6 classes
   - Python/Target uses 4 classes
   - Need to verify actual dataset

6. **❌ Wrong architecture paradigm**
   - Using global attention transformer (ViT)
   - Target uses CNN + LSTM with windowed processing
   - Missing temporal/sequential modeling component

**Status**: Slightly worse than Python implementation (~36% vs ~39%), both far below target performance

---

## Gap Analysis

### Performance Gap
- **Current**: ~36-39% accuracy, ~0.66 AUROC
- **Target**: 84.7% balanced accuracy, 0.960 AUROC
- **Gap**: ~45-49 percentage points in accuracy, ~0.30 in AUROC

### Architecture Gap

The fundamental issue is **architectural paradigm mismatch**:

| Aspect | Current (ViT) | Target (LRCN) | Impact |
|--------|---------------|---------------|--------|
| **Processing** | Global attention over all 128 lines | Rolling window of W=20 lines | High |
| **Temporal modeling** | None (position embeddings only) | LSTM over scan evolution | High |
| **CNN component** | None (direct embedding) | VGG-like feature extraction | Medium |
| **Human heuristic** | Not captured | Mimics line-by-line assessment | High |

**Key insight from Gordon et al.**: "LRCN embeds the human heuristic of assessing tips by observing line-by-line scan evolution"

The Vision Transformer approach treats the problem as static image classification with position-aware attention, while the LRCN approach treats it as a temporal sequence problem where the evolution of scanlines matters.

---

## Path Forward

### Option A: Fix Current ViT Implementation (Incremental)
**Pros**: Smaller changes, keeps existing codebase
**Cons**: Unlikely to reach target performance (different paradigm)

Immediate fixes:
1. Implement class weights properly
2. Lower learning rate to 1e-4 with warmup + scheduling
3. Add per-scanline normalization
4. Increase to 6 layers
5. Verify dataset has 4 classes

**Expected outcome**: May reach ~50-60% accuracy, still far from target

### Option B: Implement LRCN Architecture (Structural)
**Pros**: Matches proven architecture, better chance at target performance
**Cons**: Requires significant rewrite

Required changes:
1. Replace transformer with CNN (VGG-like) + LSTM
2. Implement windowed processing (W=20 scan lines)
3. Add rolling window inference
4. Keep proper normalization, class weights, LR scheduling from fixes above

**Expected outcome**: Should approach target 85% accuracy, 0.96 AUROC

### Option C: Sequential Encoder-Only Transformer (Recommended)
**Pros**:
- Natural alignment with sequential scan line acquisition
- Better than LSTM: parallel training, no vanishing gradients, richer context
- More interpretable than LSTM (attention weights visualizable)
- Flexible for variable-length sequences and real-time prediction
- Combines benefits of transformer architecture with sequential modeling

**Cons**:
- May need more data than LSTM (though likely sufficient with 433k line sequences)
- Slightly more complex than direct ViT fix

#### Architecture Design

**Core Concept**: Treat STM scanning as a sequential process where each scan line is a token, with causal masking to enable progressive prediction (mimicking real-time scanning).

**1. Tokenization Layer (1D CNN per scan line)**
```
Input: Single scan line [1 × 128 pixels]
Conv1D(kernel=7, stride=2, filters=32)
  → Conv1D(kernel=3, stride=1, filters=64)
  → Conv1D(kernel=3, stride=1, filters=128)
  → GlobalAvgPool
  → Dense(256)
Output: Token embedding [256-d]
```
**Rationale**: CNN captures local spatial features within each line (atoms, dimers, defects), similar to LRCN's CNN component but applied per-line.

**2. Positional Encoding**
```
PE_{pos,2i} = sin(pos/10000^{2i/d_model})
PE_{pos,2i+1} = cos(pos/10000^{2i/d_model})
```
**Rationale**: Line position is critical - early lines vs late lines contain different information about tip stability. Sinusoidal encoding (from "Attention is All You Need") provides smooth positional information.

**3. Transformer Encoder Stack (6 layers)**
```
for layer in 1..6:
    MultiHeadSelfAttention(
        heads=8,
        d_k=32,
        causal_mask=true  // Key: only attend to current and previous lines
    )
    → Add & LayerNorm
    → FeedForward(d_model=256, d_ff=1024, activation=GELU)
    → Add & LayerNorm
    → Dropout(0.1)
```
**Rationale**:
- **Causal masking**: Position i can only attend to positions ≤ i, mirroring actual scanning where future lines aren't visible yet
- **Self-attention**: Can capture both local (adjacent lines) and long-range dependencies (patterns across entire scan)
- **6 layers**: Matches Python ViT depth, proven sufficient for this task complexity

**4. Classification Head**
```
// Per-sequence classification (using final token)
FinalToken → LayerNorm → Dense(num_classes=4) → Softmax

// OR per-token classification (for progressive prediction)
EachToken → LayerNorm → Dense(num_classes=4) → Softmax
```
**Rationale**: Per-sequence for full-scan classification, per-token for real-time progressive assessment.

#### Why This Should Outperform LRCN

**1. Richer Context Modeling**
- **LSTM**: Hidden state h_t depends only on h_{t-1} (sequential bottleneck)
- **Transformer**: Each line directly attends to all previous lines
- Result: Can capture complex, non-local dependencies that LSTMs might miss

**2. Better Gradient Flow**
- **LSTM**: Vanishing/exploding gradient issues over 128-step sequences
- **Transformer**: Direct paths through attention mechanism
- Result: More stable training, better optimization

**3. Parallel Training**
- **LSTM**: Must process sequentially (h_1 → h_2 → ... → h_128)
- **Transformer**: All positions processed in parallel during training
- Result: ~10-100× faster training

**4. Interpretability**
- **LSTM**: Opaque hidden states, hard to debug
- **Transformer**: Attention weights show which lines the model focuses on
- Result: Can visualize *where* model detects tip changes, useful for debugging and trust

**5. Flexibility**
- **LSTM with W=20**: Fixed window, needs 20 lines minimum
- **Transformer**: Can predict after any number of lines (1, 5, 10, 20, ...)
- Result: More flexible for real-time deployment

#### Comparison Matrix

| Aspect | Current ViT | LRCN (Gordon 2020) | Sequential Transformer (Proposed) |
|--------|-------------|-------------------|----------------------------------|
| **Tokenization** | Direct linear | CNN features | 1D CNN per line |
| **Temporal modeling** | None (global attention) | LSTM (sequential) | Causal self-attention |
| **Context window** | All 128 lines | W=20 lines | Progressive (1 to 128) |
| **Training speed** | Fast (parallel) | Slow (sequential) | Fast (parallel) |
| **Gradient flow** | Good | Vanishing gradient risk | Excellent |
| **Interpretability** | Attention weights | Opaque hidden states | Attention weights (causal) |
| **Real-time prediction** | No (needs full scan) | After W=20 lines | After any line |
| **Expected AUROC** | 0.66 (achieved) | 0.960 (proven) | 0.965-0.980 (estimated) |
| **Data efficiency** | Moderate | Good | Moderate (likely sufficient) |

#### Implementation Strategy (3 Phases)

**Phase 1: Baseline Replication (Target: Match LRCN)**
- Window size W=20 lines (for fair comparison with LRCN)
- 1D CNN tokenizer (match LRCN's CNN feature extractor)
- 4-6 transformer encoder layers with causal masking
- Standard training: class weights, normalization, LR scheduling
- **Goal**: Achieve ≥0.96 AUROC with W=20

**Phase 2: Progressive Improvement (Target: Exceed LRCN)**
- Variable window sizes (W = 10, 20, 30, 40, full 128)
- Experiment with tokenization strategies:
  - 1D CNN depth (3 vs 5 layers)
  - Kernel sizes (3, 5, 7)
  - Pooling strategies (max vs avg)
- Add per-token classification for progressive prediction
- **Goal**: Match full-scan performance with fewer lines

**Phase 3: Optimization & Deployment (Target: Real-time)**
- KV-cache for efficient incremental prediction
- Quantization (FP16 or INT8) for speed
- Attention pattern analysis and visualization
- Hybrid approach: simple classifier for lines 1-10, transformer for 10+
- **Goal**: Real-time inference (<100ms per line)

#### Expected Performance

**Conservative estimate** (assuming proper implementation):
- **AUROC**: 0.960-0.970 (match or slightly exceed LRCN)
- **Balanced Accuracy**: 0.850-0.880 (match or exceed LRCN)
- **Training time**: 2-5× faster than LRCN (parallel processing)
- **Inference latency**: <100ms per prediction (with optimization)

**Optimistic estimate** (if transformer excels at this task):
- **AUROC**: 0.975-0.985 (significantly exceed LRCN)
- **Balanced Accuracy**: 0.900-0.920 (approach human-level)
- **Early prediction**: Reliable classification after 10-15 lines vs LRCN's 20

#### Advantages for STM Use Case

**Tip Change Detection**:
- Attention weight analysis reveals *exactly which lines* trigger classification changes
- Sudden attention pattern shifts → tip change detected mid-scan
- Can abort/restart scan immediately upon detection

**Multi-Resolution Understanding**:
- Lower layers capture local features (individual atoms)
- Higher layers capture global patterns (rows, overall quality)
- Natural hierarchical representation

**Transfer Learning Potential**:
- Pre-train on H:Si(100) (current dataset)
- Fine-tune on other surfaces (Au, graphene, etc.) with minimal data
- Foundation model for STM tip assessment across materials

**Anomaly Detection**:
- Attention entropy analysis: high entropy = uncertain/anomalous
- Can flag unusual patterns that don't fit training distribution
- Safer for autonomous operation

---

## Recommendation

**Implement Option C (Sequential Encoder-Only Transformer)** because:

1. **Natural alignment**: STM data is inherently sequential - transformers excel at sequence modeling
2. **Proven architecture**: Encoder-only transformers are well-established (BERT, RoBERTa)
3. **Addresses ViT failure**: Current ViT fails because it lacks sequential/causal structure
4. **Better than LSTM**: Parallel training, richer context, better gradients, interpretable
5. **Flexible deployment**: Can predict after any number of lines, not fixed window
6. **Data sufficient**: 433k line sequences should be enough for transformer training
7. **Best of both worlds**: Combines transformer's representation power with LRCN's sequential insight

The key innovation is **causal masking** - this single change transforms the ViT from a static image classifier into a sequential predictor that mimics real-time scanning. This captures Gordon et al.'s key insight (temporal evolution matters) while leveraging transformer advantages.

---

## Dataset Details

**Current data**:
- Format: 128×128 STM images (128 scanlines × 128 pixels per line)
- Stored as NPZ files
- Pre-augmented (Python version used horizontal flip, Gaussian noise, line dropout)

**Expected distribution (from Python analysis)**:
- 4 classes with severe imbalance:
  - Class 0 (atoms): 8.79%
  - Class 1 (asymmetries/dimers): 43.90%
  - Class 2 (rows): 16.91%
  - Class 3 (defects): 30.39%

**Note**: Rust model is hardcoded to 6 classes but should be 4. Dataset verification needed.

---

## Next Steps

1. Verify actual number of classes in current dataset
2. Decide between Option A (fix ViT) or Option B (implement LRCN)
3. If Option B: Design LRCN architecture in Burn framework
4. Implement windowed data loading (W=20 scanlines)
5. Add CNN feature extractor (VGG-like)
6. Add LSTM temporal modeling
7. Implement all training fixes (class weights, normalization, LR schedule)
8. Target metrics: 0.96 AUROC, 0.847 balanced accuracy
