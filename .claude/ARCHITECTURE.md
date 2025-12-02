# Architecture Plans

## Current Architecture: Vision Transformer (ViT)

**Implementation**: `src/model.rs::ScanLineEncoder`

```
Input: [batch, 128 lines, 128 pixels]
  ↓
Line Embedding: Linear(128 pixels → 256 dims)
  ↓
CLS Token Prepend: [batch, 129, 256]
  ↓
Positional Embedding: Learned embeddings [0..129]
  ↓
Transformer Encoder: 6 layers
  - Multi-head attention (8 heads, global)
  - Feed-forward (256 → 1024 → 256)
  - Pre-normalization, dropout 0.1
  ↓
CLS Token Extraction: [batch, 256]
  ↓
Classification Head: Linear(256 → 4 classes)
  ↓
Output: [batch, 4]
```

**Key Characteristics**:
- Global attention: Each line can attend to all other lines
- Position-aware but not causal
- No sequential/temporal modeling
- Processes entire scan simultaneously

---

## Proposed Architecture: Sequential Encoder-Only Transformer

**Status**: Planned (Option C from original analysis)

### Design Philosophy

Treat STM scanning as a **sequential process** where each scan line is a token, with **causal masking** to enable progressive prediction (mimicking real-time scanning).

### Architecture Components

#### 1. Tokenization Layer (1D CNN per scan line)

```
Input: Single scan line [1 × 128 pixels]
  ↓
Conv1D(kernel=7, stride=2, filters=32)
  ↓
Conv1D(kernel=3, stride=1, filters=64)
  ↓
Conv1D(kernel=3, stride=1, filters=128)
  ↓
GlobalAvgPool
  ↓
Dense(256)
  ↓
Output: Token embedding [256-d]
```

**Rationale**: CNN captures local spatial features within each line (atoms, dimers, defects), similar to LRCN's CNN component but applied per-line.

#### 2. Positional Encoding

```python
PE_{pos,2i} = sin(pos/10000^{2i/d_model})
PE_{pos,2i+1} = cos(pos/10000^{2i/d_model})
```

**Rationale**: Line position is critical - early lines vs late lines contain different information about tip stability. Sinusoidal encoding provides smooth positional information.

#### 3. Transformer Encoder Stack (6 layers)

```
for layer in 1..6:
    MultiHeadSelfAttention(
        heads=8,
        d_k=32,
        causal_mask=true  // ← KEY: only attend to current and previous lines
    )
    → Add & LayerNorm
    → FeedForward(d_model=256, d_ff=1024, activation=GELU)
    → Add & LayerNorm
    → Dropout(0.1)
```

**Rationale**:
- **Causal masking**: Position i can only attend to positions ≤ i
- **Self-attention**: Captures both local and long-range dependencies
- **6 layers**: Proven sufficient for task complexity

#### 4. Classification Head

```
// Per-sequence classification (using final token)
FinalToken → LayerNorm → Dense(num_classes=4) → Softmax

// OR per-token classification (for progressive prediction)
EachToken → LayerNorm → Dense(num_classes=4) → Softmax
```

---

## Architecture Comparison

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

---

## Implementation Strategy (3 Phases)

### Phase 1: Baseline Replication (Target: Match LRCN)
**Goal**: Achieve ≥0.96 AUROC

**Tasks**:
- [ ] Implement 1D CNN tokenizer per scanline
- [ ] Add causal attention mask to transformer
- [ ] Change positional embeddings from learned to sinusoidal
- [ ] Test with window size W=20 for fair comparison

**Expected Results**: Match or slightly exceed LRCN performance

### Phase 2: Progressive Improvement (Target: Exceed LRCN)
**Goal**: Match full-scan performance with fewer lines

**Tasks**:
- [ ] Experiment with variable window sizes
- [ ] Tune CNN tokenizer depth
- [ ] Add per-token classification for progressive prediction

**Expected Results**: Reliable classification after 10-15 lines vs LRCN's 20

### Phase 3: Optimization & Deployment (Target: Real-time)
**Goal**: Real-time inference (<100ms per line)

**Tasks**:
- [ ] Implement KV-cache for efficient incremental prediction
- [ ] Apply quantization (FP16 or INT8)
- [ ] Analyze and visualize attention patterns

**Expected Results**: <100ms inference latency
