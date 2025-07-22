# PyTorch Image Rotation

Dataset help:
- All datasets contain the original image, the modified image, the angle of rotation, and if applicable the units of translation.
    - `copy_rotated_dataset.pt` is the MNIST digit dataset, with only rotation applied.
    - `copy_rotated_translated_dataset.pt` is the MNIST digit dataset, with rotation and translation applied.
    - `copy_rotated_dataset_cifar10.pt` is a dataset with real images, with only rotation applied.

When creating a YAML file, if no dataset is specified, the default is the MNIST digit dataset with only rotation applied.


### Weights and Biases: 

Translation only:
- https://wandb.ai/mtonon-/baseline_linear_relu_regression?nw=nwusermtonon 

Translation and rotation:
- https://wandb.ai/mtonon-/translation_and_rotation?nw=nwusermtonon
