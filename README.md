# PyTorch Image Rotation

Dataset help:
- All datasets contain the original image, the modified image, and the angle of rotation.
    - `copy_rotated_dataset.pt` is the MNIST digit dataset, with only rotation applied.
    - `copy_rotated_translated_dataset_mnist.pt` is the MNIST digit dataset, with rotation and translation applied.
    - `copy_rotated_translated_dataset_cifar10.pt` is the CIFAR10 dataset with real images, with rotation and translation applied.

When creating a YAML file, if no dataset is specified, the default is the MNIST digit dataset with only rotation applied.


### Weights and Biases: 

Translation only (MNIST):
- https://wandb.ai/mtonon-/baseline_linear_relu_regression?nw=nwusermtonon 

Translation and rotation (MNIST):
- https://wandb.ai/mtonon-/translation_and_rotation_mnist?nw=nwusermtonon

Translation and rotation (CIFAR10):
- https://wandb.ai/mtonon-/translation_and_rotation_cifar?nw=nwusermtonon

