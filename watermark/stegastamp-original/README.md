---
license: apache-2.0
tags:
  - steganography
  - image-watermarking
  - pytorch
  - stegastamp
library_name: pytorch
---

# StegaStamp Original

A PyTorch implementation of StegaStamp (Tancik et al., CVPR 2020) trained on COCO dataset.

## Model Details

- **Architecture**: StegaStamp encoder-decoder with Spatial Transformer Network (STN)
- **Training**: 140,000 steps on COCO train2017 images
- **Input size**: 400×400 RGB images
- **Message capacity**: 100 bits

## Performance

| Metric | Value |
|--------|-------|
| Bit Accuracy | 99.6% |
| PSNR | 27.95 dB |

## Training Configuration

```yaml
experiment_name: stegastamp_original
data:
  batch_size: 4
training:
  num_steps: 140000
  lr: 0.0001
  stn_lr_scale: 0.01  # Critical for training stability
loss:
  message: { scale: 7.0, ramp_steps: 1 }
  l2: { scale: 2.0, ramp_steps: 10000 }
  lpips: { scale: 1.5, ramp_steps: 10000 }
distortion:
  strategy: curriculum
```

### Key Training Insight

The STN linear parameters (`stn_fc_weight`, `stn_fc_bias`) require a 100× lower learning rate to prevent training collapse. Without this, the decoder outputs constant 0.5 probabilities (loss stuck at 0.693).

## Usage

```python
import torch
from picode.models.stegastamp import Encoder, Decoder

# Load checkpoint
checkpoint = torch.load("best.pt", map_location="cpu")
encoder = Encoder(num_bits=100)
decoder = Decoder(num_bits=100)
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

# Encode message into image
image = ...  # (B, 3, 400, 400) tensor in [0, 1]
message = torch.randint(0, 2, (B, 100)).float()
encoded_image = encoder(image, message)

# Decode message from image
logits = decoder(encoded_image)
decoded_message = (torch.sigmoid(logits) > 0.5).float()
```

## Citation

```bibtex
@inproceedings{tancik2020stegastamp,
  title={StegaStamp: Invisible Hyperlinks in Physical Photographs},
  author={Tancik, Matthew and Mildenhall, Ben and Ng, Ren},
  booktitle={CVPR},
  year={2020}
}
```

## License

Apache 2.0
