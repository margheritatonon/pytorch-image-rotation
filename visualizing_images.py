import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

#visualization:
import torch
import matplotlib.pyplot as plt

#loading
#original_images, rotated_images, angles = torch.load("dataset/copy_rotated_translated_dataset_cifar10.pt")
original_images, rotated_images, angles = torch.load("dataset/copy_rotated_translated_dataset.pt")

#plotting
fig, axes = plt.subplots(2, 10, figsize=(15, 3))
axes = axes.flatten()

for i in range(10):
    #CHW (3, H, W) to HWC (H, W, 3)
    squeezed_original = original_images[i].permute(1, 2, 0).cpu().numpy()
    squeezed_rotated = rotated_images[i].permute(1, 2, 0).cpu().numpy()

    #padded images plotting
    axes[i].imshow(squeezed_original)
    axes[i].set_title(f"Original {i+1}")
    axes[i].axis('off')

    #rotated images plotting - we don't have information about the translation because in the end we still want to predict just the angle
    axes[i + 10].imshow(squeezed_rotated)
    axes[i + 10].set_title(f"Angle {angles[i].item():.0f}°")
    axes[i + 10].axis('off')

plt.tight_layout()
plt.show()


#to run:
# .venv/bin/python /Users/margheritatonon/pytorch-image-rotation/visualizing_cifar10_translatedrotated.py
