import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import config
from utils import to_grayscale

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Open image and convert to RGB
        if self.transform:
            image = self.transform(image)
        return image
    
## Testing
if __name__ == "__main__":
    ds = CustomImageDataset(image_dir="Dataset/Train", transform=config.transform)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    x = next(iter(dl))
    print(x.min(), x.max())