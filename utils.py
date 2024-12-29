import torch
import config
from torchvision.utils import save_image
import cv2
import numpy as np
import wandb
import os
import numpy as np
import ee
import cv2


def save_some_examples(gen, val_loader, epoch, folder):
    x,z1,y = next(iter(val_loader))
    x,z1,y = x.to(config.DEVICE), z1.to(config.DEVICE), y.to(config.DEVICE)
    
    gen.eval()
    with torch.no_grad():
        y_fake =  gen(x,z1=z1)
        # y_fake = (y_fake > 0.5).float() 
        # y = (y > 0.5).float()

        x = x*0.5+0.5
        y = y*0.5+0.5
        y_fake = y_fake*0.5+0.5

        stacked_images = torch.cat((x,y,y_fake), dim=2)
        save_image(stacked_images, folder + f"/y_gen_{epoch}.png")
        save_image(x, folder + f"/input_{epoch}.png")
        wandb.log({
            "Generated Images": [wandb.Image(f"/content/evaluation/y_gen_{epoch}.png", caption=f"Epoch {epoch} - Generated")]
        })
        if epoch == 1 or epoch == 0:
            save_image(y, folder + f"/label_{epoch}.png")
    gen.train()



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def to_grayscale(image_tensor: torch.Tensor):
    """
    Converts an RGB image tensor to grayscale with a single channel.
    Args:
        image_tensor (torch.Tensor): Tensor of shape (C, H, W) or (N, C, H, W)
    Returns:
        torch.Tensor: Grayscale tensor of shape (1, H, W) or (N, 1, H, W)
    """
    if len(image_tensor.shape) == 4:  # Batch of images
        r, g, b = image_tensor[:, 0, :, :], image_tensor[:, 1, :, :], image_tensor[:, 2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(1)  # Add channel dim
    
    elif len(image_tensor.shape) == 3:  # Single image
        if not isinstance(image_tensor, torch.Tensor): 
            r, g, b = image_tensor[:, :, 0], image_tensor[:, :, 1], image_tensor[:, :, 2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            return gray
        
        r, g, b = image_tensor[0, :, :], image_tensor[1, :, :], image_tensor[2, :, :]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray.unsqueeze(0)  # Add channel dim
    
    else:
        raise ValueError("Expected input to have 3 or 4 dimensions.")
    
