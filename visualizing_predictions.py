import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from src.models import ResNet  #this can change based on themodel

#loading model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ResNet(resnet_type=34, unit_norm=True)  # adjust if needed
model.load_state_dict(torch.load("checkpoints/resnet2-34.pth", map_location=device))
model.to(device)
model.eval()

#loading pt dataset
original_imgs, transformed_imgs, angles = torch.load("dataset/copy_rotated_translated_dataset_cifar10.pt") 

#visualizing
def visualize_prediction(original_img, rotated_img, pred_angle_rad, idx=None):
    pred_angle_deg = torch.rad2deg(pred_angle_rad).item()
    reconstructed_img = TF.rotate(original_img, angle=pred_angle_deg, interpolation=TF.InterpolationMode.BILINEAR)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(original_img.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray' if original_img.shape[0] == 1 else None)
    axs[0].set_title("Original")

    axs[1].imshow(rotated_img.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray' if original_img.shape[0] == 1 else None)
    axs[1].set_title("Rotated Input")

    axs[2].imshow(reconstructed_img.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray' if original_img.shape[0] == 1 else None)
    axs[2].set_title(f"Predicted: {pred_angle_deg:.2f}°")

    for ax in axs:
        ax.axis("off")
    if idx is not None:
        plt.suptitle(f"Sample {idx}")
    plt.tight_layout()
    plt.show()


with torch.no_grad():
    for i in range(min(5, original_imgs.size(0))):
        original = original_imgs[i].unsqueeze(0).to(device)  # shape (1, C, H, W)
        rotated = transformed_imgs[i].unsqueeze(0).to(device)

        inputs = torch.cat((original, rotated), dim=1)

        pred = model(inputs)
        pred_sin, pred_cos = pred[:, 0], pred[:, 1]
        pred_angle_rad = torch.atan2(pred_sin, pred_cos)

        # Visualize
        visualize_prediction(original[0].cpu(), rotated[0].cpu(), pred_angle_rad[0].cpu(), idx=i)
