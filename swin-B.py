import os
import torch
from torch.utils.data import Dataset
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import segmentation_models_pytorch as smp
from pprint import pprint
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from PIL import Image
import evaluate
from SwinBtransformer import SwinTransformerSys
import cv2

from sklearn.metrics import confusion_matrix
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from sklearn.metrics import classification_report

METAINFO = {
    "classes": (
        "unlabelled", "dirt", "mud", "water", "gravel", "other-terrain",
        "tree-trunk", "tree-foliage", "bush", "fence", "structure", "other-object",
        "rock", "log", "sky", "grass"
    ),
    "cidx": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18],
    "palette": [
        (0, 0, 0),
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 190),
        (0, 128, 128),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
        (128, 128, 0),
        (255, 215, 180),
        (0, 0, 128),
        (128, 128, 128),
    ],
}

class_mapping = {
    0: 0,   # unlabelled -> unlabelled
    1: 6,   # asphalt -> other-terrain (6)
    2: 1,   # dirt -> dirt (1)
    3: 2,   # mud -> mud (2)
    4: 3,   # water -> water (3)
    5: 4,   # gravel -> gravel (4)
    6: 5,   # other-terrain -> other-terrain (5)
    7: 6,   # tree-trunk -> tree-trunk (6)
    8: 7,   # tree-foliage -> tree-foliage (7)
    9: 8,   # bush -> bush (8)
    10: 9,  # fence -> fence (9)
    11: 10, # structure -> structure (10)
    12: 11, # pole -> other-object (11)
    13: -1, # vehicle -> exclude
    14: 12, # rock -> rock (12)
    15: 13, # log -> log (13)
    16: 11, # other-object -> other-object (11)
    17: 14, # sky -> sky (14)
    18: 15  # grass -> grass (15)
}

num_classes = 16
class BCdataset(Dataset):
    def __init__(self, df_data, class_mapping):
        self.ori_img_path_list = list(df_data['ori_img'])
        self.mask_img_path_list = list(df_data['mask_img'])
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.ori_img_path_list)

    def __getitem__(self, idx):
        image_path = self.ori_img_path_list[idx]
        mask_path = self.mask_img_path_list[idx]

        image = np.array(Image.open(image_path).resize((256, 256)).convert("RGB"))
        mask = np.array(Image.open(mask_path).resize((256, 256))).astype('int32')
        # Create a new mask to ensure unmapped values are handled
        mapped_mask = np.zeros_like(mask)

        # Apply the class mapping
        for original_class, new_class in self.class_mapping.items():
            mapped_mask[mask == original_class] = new_class

        mapped_mask[mapped_mask == -1] = 0
        # Ensure all values are mapped
        unique_values = np.unique(mapped_mask)
        for value in unique_values:
            if value not in self.class_mapping.values():
                print(f"Warning: Unmapped label found: {value}")

        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image).float()
        mapped_mask = torch.tensor(mapped_mask).long()

        return image, mapped_mask


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for (img_rgb, y) in dataloader:
            img_rgb = img_rgb.to(device)

            y = y.to(device)

            pred = model(img_rgb)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n , Avg loss: {test_loss:>8f} \n")
    return test_loss

def get_metrics(model, test_dl):
  model.eval()
  label_list = []
  pred_list = []
  with torch.no_grad():
      for images, labels in test_dl:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model(images)
          _, preds = torch.max(outputs, 1)
          preds_l = list(preds.cpu().numpy())
          labels_l = list(labels.cpu().numpy())
          label_list.append(labels.cpu().numpy().flatten())
          pred_list.append(preds.cpu().numpy().flatten())
          mean_iou.add_batch(predictions=preds_l, references=labels_l)

  label_array = np.concatenate(label_list, axis=0)
  pred_array = np.concatenate(pred_list, axis=0)

  classify_result = classification_report(label_array,pred_array,digits=3)
  print(classify_result)


  iou_result = mean_iou.compute( num_labels=num_classes, ignore_index=None)
  print(iou_result)

  return classify_result,  iou_result

def train_test_plot(train_losses, test_losses):
    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.savefig('loss_plot_swin_b.png')

data_dir = 'D:\9517Project\wildScene2d'
image_dir = os.path.join(data_dir, 'image')
index_label_dir = os.path.join(data_dir, 'indexLabel')

img_path_list = [str(item) for item in Path(image_dir).rglob('*.png')]
mask_path_list = [str(item) for item in Path(index_label_dir).rglob('*.png')]

img_path_list.sort()
mask_path_list.sort()

place_name_list = [item.split('/')[0] for item in img_path_list]

data_dict = {'ori_img':img_path_list,"mask_img":mask_path_list}

np.array(Image.open(mask_path_list[0]))

pd.set_option('display.max_colwidth', 500)
pd_data = pd.DataFrame(data_dict)

train_data, test_data = train_test_split(pd_data, test_size=0.2, random_state=42)  # 0.25 x 0.8 = 0.2

train_dataset = BCdataset(train_data, class_mapping)
test_dataset = BCdataset(test_data, class_mapping)
n_cpu = os.cpu_count()
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=n_cpu, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=n_cpu, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

mean_iou = evaluate.load("mean_iou")

model = SwinTransformerSys(img_size=256, num_classes=16, window_size=8).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 8
train_loss_list = []
test_loss_list = []
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_loader, model, loss_fn, optimizer)

    train_loss = test(train_loader, model, loss_fn)
    test_loss = test(test_loader, model, loss_fn)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
print("Done!")

train_test_plot(train_loss_list, test_loss_list)
classify_result,  iou_result = get_metrics(model, test_loader)

b_img,b_mask = next(iter(test_loader))
with torch.no_grad():
    model.eval()
    logits = model(b_img.to(device))
predictions = torch.argmax(logits, dim=1)

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)
for i, (image, gt_mask, pr_mask) in enumerate(zip(b_img, b_mask, predictions)):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image.long().numpy().transpose(1, 2, 0))  # convert CHW -> HWC
    plt.title("Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(gt_mask.numpy().squeeze())  # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(pr_mask.cpu().numpy().squeeze())  # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")
    output_path = os.path.join(output_dir, f"comparison_{i}.png")
    plt.savefig(output_path)
    plt.close()
print(f"Saved {len(b_img)} images to {output_dir}")