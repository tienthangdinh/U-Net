import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index]) #https://docs.python.org/3/library/os.path.html
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        #read the images and convert to numpy array
        image = np.array(Image.open(img_path).convert("RGB")) #nunpy.array https://numpy.org/doc/2.1/reference/generated/numpy.array.html
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) #https://pillow.readthedocs.io/en/stable/reference/Image.html
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask) #using albumentations transform here instead of pytorch because pytorch can only transform image. but albumentation can transform both input and output
            image = augmentations["image"]
            mask = augmentations["mask"] #normally no need to transform this

        return image, mask
