import torch
from torchvision import transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160  # 1280 originally
IMAGE_WIDTH = 240  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/train_images/"
VAL_MASK_DIR = "data/train_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader) # just need to pass an iterable inside, return another decorated iterable https://tqdm.github.io/docs/tqdm/

    for batch_idx, (data, targets) in enumerate(loop): #tqdm(dataloader) is ofcourse a decorated iterable
        data = data.to(device=DEVICE) #load data to cuda
        targets = targets.float().unsqueeze(1).to(device=DEVICE)  #add a new dimension after the batch and wrapping around the image, then load to device

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data) #forward first
            loss = loss_fn(predictions, targets) #loss(predictions, targets)

        # backward
        optimizer.zero_grad() #delete all gradient
        scaler.scale(loss).backward() #backward
        scaler.step(optimizer)
        scaler.update() 

        # update tqdm loop
        loop.set_postfix(loss=loss.item()) #set additional stats https://tqdm.github.io/docs/tqdm/#set_postfix


def main():

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)), #https://pytorch.org/vision/0.9/transforms.html
        transforms.RandomRotation(degree=35),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor(), #convert a PIL Image or numpy.ndarray => Tensor (C,H,W) (the input image is scaled to [0 - 1]) => should not use this when transforming target image mask
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #apply 3 values for each channel as the first dimension (we know color channel, but for pytorch its simply assume the first dimension)
    ])


    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE) #load model to device
    loss_fn = nn.BCEWithLogitsLoss() #u-net is also actually a binary classification task
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler() #FP16 vs FP32 torch.amp

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()