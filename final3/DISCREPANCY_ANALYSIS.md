# Discrepancy Analysis: training_strategy.md vs Current Implementation

## Critical Mismatches Found ⚠️

### 1. Vision Encoder Model Variant
| Source | Specification |
|--------|--------------|
| **training_strategy.md** | DINOv3 ViT-B/**16** (16×16 patch size) |
| **Current Implementation** | DINOv3 ViT-B/**14** (14×14 patch size) |
| **instructions.txt** | "DINOv3 ViT-B" (no patch size specified) |

**Impact**: Different model architecture, different feature granularity
**Action Needed**: Which model should we use?

---

### 2. Embedding Dimension & Projection Strategy
| Source | Vision→Text Projection |
|--------|----------------------|
| **training_strategy.md** | Vision 768 → **512** (match text side) |
| **Current Implementation** | Vision 768 → 1536 → **768**, Text 512 → 1536 → **768** |

**Impact**: Completely different shared embedding space
**Action Needed**: Should we project to 512-dim or 768-dim?

---

### 3. Total Training Steps
| Source | Total Steps | Reasoning |
|--------|------------|-----------|
| **training_strategy.md** | **25,000 steps** | "Derived from PLIP paper" |
| **Current Implementation** | **~115,549 steps** | 4 epochs × (462,195/16) |
| **instructions.txt** | Not specified | Says "4 epochs" from CheXzero |

**Impact**: 4.6× difference in training duration!
**Action Needed**: How many steps total?

---

### 4. Batch Size
| Source | Batch Size |
|--------|-----------|
| **training_strategy.md** | **64** |
| **Current Implementation** | **16** |
| **CheXzero (run_train.py:23)** | **16** |

**Note**: training_strategy.md says "Strictly following the CheXzero batch size" but CheXzero uses 16, not 64!
**Action Needed**: Which batch size is correct?

---

### 5. Validation Frequency
| Source | Eval Interval |
|--------|--------------|
| **training_strategy.md** | Every **500 steps** |
| **Current Implementation** | Every **50 steps** |
| **instructions.txt** | "validate very frequently" (PLIP strategy) |

**Impact**: 10× difference in validation frequency
**Action Needed**: How often should we validate?

---

### 6. Text Context Length
| Source | Context Length |
|--------|---------------|
| **training_strategy.md** | **512 tokens** ("distilled context for long medical reports") |
| **Current Implementation** | **77 tokens** |
| **CheXzero (run_train.py:32)** | **77 tokens** |
| **CLIP Default** | **77 tokens** |

**Impact**: 512 tokens requires custom text encoder configuration
**Action Needed**: Can CLIP text encoder handle 512 tokens? Standard is 77.

---

### 7. Normalization Statistics
| Source | Mean | Std |
|--------|------|-----|
| **training_strategy.md** | [0.481, 0.457, 0.408] | [0.268, 0.261, 0.275] |
| **Current Implementation** | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] |

**Note**:
- training_strategy.md says "Specific to OpenAI CLIP weights"
- Current uses ImageNet stats (standard for DINOv3)
- CheXzero uses different stats: mean=101.48761, std=83.43944 (see train.py:85)

**Action Needed**: Which normalization should we use?

---

### 8. Data Augmentation
| Source | Augmentations |
|--------|--------------|
| **training_strategy.md** | Random Crop, Random Flip |
| **Current Implementation** | None (only Resize + Normalize) |
| **CheXzero** | None (only Resize + Normalize) |

**Impact**: Augmentations can help prevent overfitting
**Action Needed**: Should we add augmentations?

---

### 9. Save Checkpoint Frequency
| Source | Save Interval |
|--------|--------------|
| **training_strategy.md** | Not specified |
| **Current Implementation** | Every 100 steps |

**Impact**: Minor - can adjust based on total steps

---

## Summary Table

| Parameter | training_strategy.md | Current Implementation | CheXzero Reference | instructions.txt |
|-----------|---------------------|----------------------|-------------------|-----------------|
| Vision Model | ViT-B/16 | ViT-B/14 | N/A | "ViT-B" (unclear) |
| Embed Dim | 512 | 768 | 512 | Not specified |
| Total Steps | 25,000 | ~115,549 | Variable | Not specified |
| Batch Size | 64 | 16 | **16** | Not specified |
| Epochs | N/A | 4 | **4** | Not specified |
| Eval Interval | 500 | 50 | By epoch | "very frequently" |
| Text Context | 512 | 77 | **77** | Not specified |
| Normalize Mean | CLIP stats | ImageNet | CheXzero custom | Not specified |
| Normalize Std | CLIP stats | ImageNet | CheXzero custom | Not specified |
| Augmentations | Crop+Flip | None | None | Not specified |

---

## Conflicting Requirements

### Batch Size Contradiction
training_strategy.md states:
> "Batch Size: 64 - Strictly following the CheXzero batch size"

But CheXzero actually uses **batch_size=16** (run_train.py:23)

This is a **factual error** in training_strategy.md.

### Context Length Issue
training_strategy.md requests:
> "Text Context Length: 512 tokens - Uses the distilled context length for long medical reports"

But:
- CLIP text encoder has a **maximum of 77 tokens** by architecture
- CheXzero uses 77 tokens
- Changing to 512 would require retraining the text encoder or using a different architecture

This may be **technically infeasible** without major changes.

---

## Recommendations

### Option A: Follow training_strategy.md Exactly
**Changes Required:**
1. ✅ Switch to DINOv3 ViT-B/16 (from ViT-B/14)
2. ✅ Change projection to 512-dim shared space
3. ✅ Train for 25,000 steps total
4. ⚠️ Batch size 64 (contradicts CheXzero=16)
5. ✅ Validate every 500 steps
6. ❌ **Cannot** use 512 token context (CLIP limit is 77)
7. ✅ Use CLIP normalization stats
8. ✅ Add Random Crop + Flip augmentations

**Concerns:**
- 512 token context is impossible without changing text encoder
- Batch size 64 contradicts the document's own claim about "following CheXzero"

### Option B: Follow instructions.txt + CheXzero
**Current Implementation:**
1. ✅ DINOv3 ViT-B as vision encoder (we chose /14, could be /16)
2. ✅ CheXzero text encoder (CLIP)
3. ✅ CheXzero hyperparameters (batch=16, epochs=4, lr=1e-4, SGD)
4. ✅ PLIP strategy (frequent validation by steps)
5. ❌ Missing augmentations

**Advantages:**
- Matches CheXzero proven approach
- Technically feasible
- Consistent with instructions.txt

### Option C: Hybrid Approach
**Recommended Changes:**
1. Clarify ViT-B/14 vs ViT-B/16
2. Use 512-dim embedding space (training_strategy.md)
3. Keep batch_size=16 (CheXzero, technically correct)
4. Calculate total steps: if training for 25,000 steps, how many epochs is that?
   - 25,000 steps ÷ (462,195/16 per epoch) ≈ 0.86 epochs
   - This seems very short!
5. Keep context_length=77 (CLIP architectural limit)
6. Add augmentations (Random Crop, Flip)
7. Use CLIP normalization stats

---

## Questions for Clarification

1. **Which is authoritative**: training_strategy.md or instructions.txt + CheXzero?
2. **Vision model**: ViT-B/14 or ViT-B/16?
3. **Embedding dimension**: 512 or 768?
4. **Training duration**: 25,000 steps (< 1 epoch) or 4 epochs (~115k steps)?
5. **Batch size**: 16 (CheXzero) or 64 (training_strategy.md)?
6. **Text context**: Must be 77 (CLIP limit) - can we override the 512 request?
7. **Should we add augmentations** (Crop, Flip)?

Without clarification, we risk building the wrong model!
