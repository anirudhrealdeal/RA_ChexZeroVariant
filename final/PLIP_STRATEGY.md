# PLIP Training Strategy Explained

## What is PLIP?

**PLIP** (Pathology Language-Image Pre-training) is a vision-language foundation model for medical imaging that adapts CLIP's contrastive learning approach for medical data.

## Key Differences from Original CheXzero

### 1. Vision Encoder

**Original CheXzero:**
- Uses CLIP's ViT-B/32 vision encoder (pretrained on natural images)
- 224x224 resolution
- ViT with patch size 32

**Our PLIP Variant:**
- Uses **DINOv2 ViT-B/14** (stronger self-supervised pretrained model)
- 320x320 resolution (higher detail)
- ViT with patch size 14 (finer-grained features)
- DINOv2 has shown superior performance on medical imaging tasks

**Why DINOv2?**
- Better feature quality from self-supervised learning
- More robust to domain shift (natural → medical images)
- Smaller patch size captures finer anatomical details
- State-of-the-art performance on various vision benchmarks

### 2. Training Objective

**Both use contrastive learning:**

The core idea: Pull together embeddings of matching image-text pairs, push apart non-matching pairs.

```
Given a batch of (image, text) pairs:
1. Encode all images → image_embeddings (B, D)
2. Encode all texts → text_embeddings (B, D)
3. Compute similarity matrix: logits = scale * (img_emb @ txt_emb.T)
4. Loss = CrossEntropy(logits, diagonal_labels)
   - Diagonal elements = positive pairs (matching image-text)
   - Off-diagonal = negative pairs (non-matching)
```

**Symmetric loss:**
```python
loss_i2t = CrossEntropy(image→text logits, labels)  # Image predicts text
loss_t2i = CrossEntropy(text→image logits, labels)  # Text predicts image
total_loss = (loss_i2t + loss_t2i) / 2
```

### 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Input Data                               │
│  Image (320x320x3)              Text (Impression)            │
└───────────┬────────────────────────────┬─────────────────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐    ┌──────────────────────┐
│   DINOv2 ViT-B/14     │    │  CLIP Text Encoder   │
│   (Facebook AI)       │    │  (OpenAI)            │
│   Output: 768-dim     │    │  Output: 512-dim     │
└───────────┬───────────┘    └──────────┬───────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐    ┌──────────────────────┐
│  Vision Projection    │    │  Text Projection     │
│  768 → 1024 → 512     │    │  (Identity or MLP)   │
│  + GELU + Dropout     │    │                      │
└───────────┬───────────┘    └──────────┬───────────┘
            │                            │
            ▼                            ▼
┌───────────────────────┐    ┌──────────────────────┐
│   L2 Normalize        │    │   L2 Normalize       │
└───────────┬───────────┘    └──────────┬───────────┘
            │                            │
            └────────────┬───────────────┘
                         ▼
            ┌────────────────────────┐
            │  Contrastive Loss      │
            │  (Symmetric CE)        │
            └────────────────────────┘
```

## Training Details

### Data Augmentation
Currently using:
- Resize to 320x320 with aspect ratio preservation
- Zero-padding for non-square images
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Potential additions:**
- Random horizontal flip (for PA/AP views)
- Slight rotation (±5 degrees)
- Contrast adjustment
- Gaussian blur

### Batch Size Strategy

Contrastive learning benefits from **larger batch sizes**:
- Each batch provides (B² - B) negative pairs
- Batch size 128 → 16,256 negative pairs per sample
- Batch size 256 → 65,280 negative pairs per sample

**Recommendation:**
- Start with batch_size=128 (fits on most GPUs)
- Scale to 256 or 512 if GPU memory allows
- Use gradient accumulation if needed: `effective_batch_size = batch_size × accumulation_steps`

### Learning Rate Scheduling

```python
# Warmup phase (epoch 1)
lr = base_lr * (current_step / warmup_steps)

# Main training (epochs 2-50)
lr = base_lr

# Optional: Cosine decay
lr = base_lr * 0.5 * (1 + cos(π * epoch / total_epochs))
```

### Temperature Parameter

The learnable `logit_scale` (initialized to log(1/0.07) ≈ 2.66) controls:
- How "peaked" the similarity distribution is
- Higher scale → sharper distinctions between positive and negative pairs
- Lower scale → softer similarities

```python
logits = exp(logit_scale) * (image_emb @ text_emb.T)
```

The model learns the optimal scale during training!

## Training Recipe

### Phase 1: Initial Training (Epochs 1-20)
- **Goal**: Learn basic image-text alignment
- **LR**: 1e-4
- **Batch size**: 128
- **Expected**: Loss drops quickly from ~7 to ~2

### Phase 2: Refinement (Epochs 20-40)
- **Goal**: Fine-tune representations
- **LR**: 5e-5 (reduce by half)
- **Batch size**: Same or increase to 256
- **Expected**: Loss slowly decreases to ~1.5

### Phase 3: Convergence (Epochs 40-50)
- **Goal**: Final optimization
- **LR**: 1e-5
- **Expected**: Loss stabilizes around ~1.0-1.5

## Expected Performance

### Loss Values
- **Initial**: ~7-8 (random embeddings)
- **After 10 epochs**: ~3-4
- **After 30 epochs**: ~1.5-2.5
- **Converged**: ~1.0-1.5

Lower loss = better image-text alignment!

### Validation Metrics

Monitor:
1. **Validation Loss**: Should decrease and stabilize
2. **Image→Text Retrieval**: Top-1, Top-5 accuracy
3. **Text→Image Retrieval**: Top-1, Top-5 accuracy

**Good performance:**
- Top-1 retrieval: >30% (in-batch)
- Top-5 retrieval: >60% (in-batch)

## Zero-Shot Evaluation (After Training)

Once trained, use for:

### 1. Classification
```python
# Define class prompts
prompts = [
    "chest x-ray showing pneumonia",
    "chest x-ray showing cardiomegaly",
    "normal chest x-ray"
]

# Encode prompts
text_embeddings = model.encode_text(tokenize(prompts))

# Classify image
image_embedding = model.encode_image(image)
similarities = image_embedding @ text_embeddings.T
predicted_class = similarities.argmax()
```

### 2. Retrieval
```python
# Given a query image, find similar reports
image_embedding = model.encode_image(query_image)
text_embeddings = model.encode_text(all_reports)
similarities = image_embedding @ text_embeddings.T
top_k_reports = similarities.topk(k=5)
```

### 3. Embedding Analysis
```python
# Extract embeddings for visualization
image_embeddings = []
for image in dataset:
    emb = model.encode_image(image)
    image_embeddings.append(emb)

# Visualize with t-SNE or UMAP
from sklearn.manifold import TSNE
embeddings_2d = TSNE(n_components=2).fit_transform(image_embeddings)
```

## Advantages of This Approach

1. **DINOv2 Features**
   - Better generalization to medical images
   - Stronger low-level feature extraction
   - More robust to image quality variations

2. **Large-Scale Training**
   - 462K image-text pairs (CheXpert + ReXGradient)
   - Diverse impressions and findings
   - Better coverage of pathologies

3. **Flexible Embeddings**
   - 512-dim embeddings work for many downstream tasks
   - Easy to use for retrieval, classification, clustering
   - Compatible with existing CLIP-based tools

4. **Memory Efficient**
   - Chunked HDF5 processing
   - Mixed precision training (AMP)
   - Gradient checkpointing available if needed

## Limitations & Future Work

### Current Limitations
1. Only frontal view images (PA/AP)
2. Single resolution (320x320)
3. No multi-modal fusion (e.g., patient history)
4. Text limited to impressions (not full reports)

### Future Improvements
1. **Multi-view learning**: Combine frontal + lateral views
2. **Hierarchical text**: Use full radiology reports with section headers
3. **Curriculum learning**: Start with easy pairs, gradually add harder ones
4. **Hard negative mining**: Focus on confusing image-text pairs
5. **Data augmentation**: More aggressive augmentations for robustness
6. **Larger models**: Scale to ViT-L or ViT-H for better performance

## References

- **DINOv2**: [https://arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
- **CLIP**: [https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)
- **PLIP**: [https://arxiv.org/abs/2305.01689](https://arxiv.org/abs/2305.01689)
- **CheXzero**: [https://arxiv.org/abs/2102.11467](https://arxiv.org/abs/2102.11467)
