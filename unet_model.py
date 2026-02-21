import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2
import os
from torch.utils.data.dataset import Dataset
import io
from unet_architecture import UNet



def single_image_inference(model_pth, image_input, device, save_path="output.jpg"):
   
    if hasattr(image_input, 'read'):
        img_pil = Image.open(image_input)
    elif isinstance(image_input, str):  
        img_pil = Image.open(image_input)
    elif isinstance(image_input, Image.Image): 
        img_pil = image_input
    else:
        raise ValueError("Unsupported image input type")

    if img_pil.mode == 'RGBA':
        img_pil = img_pil.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(img_pil).float().to(device)
    img = img.unsqueeze(0)

    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pth, map_location=torch.device(device)))

    pred_mask = model(img)

    img = img.squeeze(0).cpu().detach()
    img = img.permute(1, 2, 0)

    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)
    pred_mask[pred_mask < 0] = 0
    pred_mask[pred_mask > 0] = 1

    if pred_mask.shape[-1] == 1:
        pred_mask = torch.cat((pred_mask, pred_mask, pred_mask), dim=-1)

    pred_mask_np = pred_mask.numpy()
    resized_mask = cv2.resize(pred_mask_np, (640, 640), interpolation=cv2.INTER_NEAREST)

    fig = plt.figure()
    for i in range(1, 3):
        fig.add_subplot(1, 2, i)
        if i == 1:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(pred_mask, cmap="gray")

    plt.savefig(save_path)
    plt.close(fig) 

    return resized_mask  

if __name__ == "__main__":
    MODEL_PATH = "C:/Users/HP/OneDrive/Desktop/finall/models/unet_test1_15.pth"
    IMAGE_PATH = "C:/Users/HP/OneDrive/Desktop/finall/sample_data/rino.jpg"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_pil = Image.open(IMAGE_PATH)
    single_image_inference(MODEL_PATH, img_pil, device, "output.jpg")