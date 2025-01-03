import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config
from albumentations.core.transforms_interface import BasicTransform
from torchvision import transforms
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "Dataset/Train"
VAL_DIR = "Dataset/Test"
LEARNING_RATE = 2e-4
BATCH_SIZE = 10
VAL_BATCH_SIZE = 5
NUM_WORKERS = 2
IMAGE_SIZE = 32
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc_LSM_v1.pth.tar"
CHECKPOINT_GEN = "gen_LSM_v1.pth.tar"

RESIZE = None
if RESIZE != None:
    IMAGE_RESIZED = RESIZE * IMAGE_SIZE
else:
    IMAGE_RESIZED = IMAGE_SIZE
 
# Stlye loss
ALPHA = 1
BETA = 0.01

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)), # Resize the input image
    transforms.ToTensor(), # Convert to torch tensor (scales data into [0,1])
    transforms.Lambda(lambda t: (t * 2) - 1), # Scale data between [-1, 1] 
])

reverse_transform = transforms.Compose([
    transforms.Lambda(lambda t: (t + 1) / 2), # Scale data between [0,1]
    transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
    transforms.Lambda(lambda t: t * 255.), # Scale data between [0.,255.]
    transforms.Lambda(lambda t: t.cpu().numpy().astype(np.uint8)), # Convert into an uint8 numpy array
    transforms.ToPILImage(), # Convert to PIL image
])


class ThresholdTransform(BasicTransform):
    def __init__(self, thr_255, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.thr = thr_255 / 255.0  # Convert threshold to [0, 1] range

    def apply(self, img, **params):
        # Binarize image based on the threshold
        return (img > self.thr).astype(img.dtype)

    @property
    def targets(self):
        # Specify that this transform applies to the 'mask'
        return {"image": self.apply, "mask": self.apply}

transform_only_input = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED,),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_inter = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ToTensorV2(),
    ]
)

transform_only_mask_binarize = A.Compose(
    [
        A.Resize(width=IMAGE_RESIZED, height=IMAGE_RESIZED),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
        ThresholdTransform(thr_255=100),
        ToTensorV2(),
    ]
)