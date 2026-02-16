# Training Configuration Summary

## Architecture

**Vision Encoder**: Pretrained DINOv3 ViT-B/14
- Pretrained self-supervised model from Facebook AI (ImageNet pretraining)
- **Input**: 224×224 RGB images (resized from 320×320 HDF5 storage on-the-fly)
- **Output**: 768-dim features
- **Rationale**: Stronger than CLIP's vision encoder for medical imaging, finer-grained features (patch size 14 vs 32)

**Text Encoder**: Pretrained CLIP Text Encoder (from OpenAI CLIP ViT-B/32)
- Pretrained on 400M image-text pairs
- **Output**: 512-dim features
- Proven effective for medical text understanding

**Projection Heads**: Both modalities projected to 768-dim shared embedding space
- Vision: 768 → 1536 (GELU) → 768
- Text: 512 → 1536 (GELU) → 768
- L2 normalization for contrastive learning

## Hyperparameters (from CheXzero)

Based on the original CheXzero implementation:

| Parameter | Value | Source |
|-----------|-------|--------|
| `batch_size` | 16 | CheXzero (line 23 in run_train.py) |
| `num_epochs` | 4 | CheXzero (line 24) |
| `lr` | 1e-4 | CheXzero (line 25) |
| `optimizer` | SGD | CheXzero (line 30) |
| `momentum` | 0.9 | CheXzero (line 31) |
| `context_length` | 77 | CheXzero (line 32) |
| `embed_dim` | 768 | DINOv3 ViT-B output dimension |

## PLIP Strategy

Following the PLIP paper (Nature Medicine 2023), we implement:

### 1. Frequent Validation
- **Validate every 50 steps** (not every epoch)
- **Why**: Catch overfitting early - medical models tend to overfit quickly
- Models often start overfitting after just a few epochs on medical data

### 2. Save by Steps
- **Save checkpoints every 100 steps**
- **Why**: Capture model state before overfitting occurs
- More granular than epoch-based saving

### 3. Best Model Selection
- Automatically save best model based on validation loss
- Important since training is short (4 epochs) and overfitting happens fast

## Training Strategy

### Why 4 Epochs?
CheXzero found that:
- More epochs lead to overfitting on medical imaging tasks
- Vision models pretrained on natural images transfer well with minimal fine-tuning
- 4 epochs provides good balance

### Why SGD instead of AdamW?
- SGD with momentum is more stable for fine-tuning pretrained models
- Less prone to overfitting compared to adaptive methods
- CheXzero's empirical choice

### Why Batch Size 16?
- Works well for contrastive learning on medical images
- Provides enough negative examples per batch (15 negatives per positive)
- Fits comfortably in GPU memory

## Dataset

- **Training**: CheXpert-Plus (223K) + ReXGradient (239K) = 462K image-text pairs
- **Validation**: CheXpert-Plus validation set (234 images)
- **Storage**: 320×320 grayscale in HDF5 (efficient storage format)
- **Model Input**: Resized to 224×224 RGB during training (on-the-fly transformation)
  - Grayscale → RGB: Replicate single channel 3 times (standard practice for pretrained models)
  - Resize: BICUBIC interpolation (320×320 → 224×224)
- **Normalization**: ImageNet stats (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Why this works**: Resize happens during data loading (CPU) while GPU trains, negligible overhead (~1-2ms per image)

## Expected Training Time

With 1 GPU (V100/A100):
- Total steps: (462,195 / 16) × 4 epochs ≈ 115,549 steps
- Validation: Every 50 steps ≈ 2,311 validations
- Checkpoints: Every 100 steps ≈ 1,156 checkpoints saved
- **Estimated time**: 20-30 hours (depending on GPU)

## Key Differences from Original CheXzero

| Aspect | CheXzero | Our Implementation |
|--------|----------|-------------------|
| Vision Encoder | Pretrained CLIP ViT-B/32 | **Pretrained DINOv3 ViT-B/14** |
| Vision Pretraining | CLIP (400M image-text pairs) | **DINOv3 SSL (ImageNet)** |
| Input Resolution | 224×224 (resized from 320×320) | **224×224 (resized from 320×320)** ✓ Same |
| Dataset | CheXpert only (~200K) | **CheXpert + ReXGradient (462K)** |
| Validation Strategy | By epoch (~14K steps) | **By steps every 50 (PLIP)** |
| Checkpointing | By epoch | **By steps every 100 (PLIP)** |
| Embedding Dim | 512 (CLIP) | **768** (DINOv3 output) |
| Training Mode | Fine-tune pretrained CLIP | **Fine-tune pretrained DINOv3** ✓ Same approach |

**Note**: Both CheXzero and our implementation use **pretrained** encoders and fine-tune them on chest X-rays. We DO NOT train from scratch.

## Resolution Handling (320×320 vs 224×224)

**Common Confusion**: Why store at 320×320 but train at 224×224?

**Answer**: This is standard practice in deep learning, exactly as CheXzero does:

1. **Storage (HDF5)**: 320×320 grayscale
   - Original CheXpert images are high resolution
   - 320×320 preserves reasonable detail while being compact
   - Flexible for different model architectures

2. **Training (Model Input)**: 224×224 RGB
   - Pretrained models (CLIP, DINOv3) expect 224×224 input
   - Resize happens **on-the-fly** during data loading (DataLoader transform)
   - CPU performs resize while GPU trains (parallel, no bottleneck)
   - BICUBIC interpolation for smooth downsampling

3. **Performance Impact**: Negligible
   - Resize: ~1-2ms per image (CPU)
   - Forward/backward pass: ~100-200ms per batch (GPU)
   - DataLoader prefetches and transforms in parallel
   - Standard in PyTorch pipelines (ImageNet, COCO, etc.)

**CheXzero does exactly the same** (see [train.py:82-87](../train.py#L82-L87)):
```python
if pretrained:
    input_resolution = 224
    transform = Compose([
        Normalize(...),
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    ])
```

## Why DINOv3 Over CLIP Vision?

1. **Better self-supervised pretraining**: DINOv3 uses more sophisticated SSL (ImageNet-22k)
2. **Finer-grained features**: Patch size 14 vs 32 (better spatial resolution)
3. **State-of-the-art performance**: Better on medical imaging benchmarks
4. **Same input resolution**: Both expect 224×224 from pretrained weights

## Monitoring Training

Watch for:
1. **Validation loss decreasing** in first 1-2 epochs
2. **Validation loss stabilizing** around epoch 2-3
3. **Potential overfitting** after epoch 3 (val loss increases)
4. **Best model saved early** (often before final epoch)

Typical loss trajectory:
```
Initial:  ~7-8 (random embeddings)
After 1 epoch: ~3-4
After 2 epochs: ~2-3
After 3 epochs: ~1.5-2.5
After 4 epochs: ~1-2
```

## References

- **CheXzero**: Original implementation hyperparameters
- **PLIP**: Nature Medicine 2023 - validation & checkpointing strategy
- **DINOv3**: Facebook AI Research - vision encoder
- **Instructions**: DINOv3 ViT-B + PLIP strategy + CheXzero hyperparameters
