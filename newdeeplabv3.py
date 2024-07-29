import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np

METAINFO = {
    "classes": (
        "unlabelled", "dirt", "mud", "water", "gravel", "other-terrain",
        "tree-trunk", "tree-foliage", "bush", "fence", "structure", "other-object",
        "rock", "log", "sky", "grass"
    ),
    "cidx": [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 17, 18]
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

def process_label(label, class_mapping):
    label_np = np.array(label)
    processed_label_np = np.zeros_like(label_np)
    for orig_class, new_class in class_mapping.items():
        if new_class == -1:
            continue
        processed_label_np[label_np == orig_class] = new_class
    return Image.fromarray(processed_label_np)

class CustomDataset(Dataset):
    def __init__(self, root, split, class_mapping, transform_image=None, transform_mask=None):
        self.root = root
        self.split = split
        self.class_mapping = class_mapping
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.images_dir = os.path.join(root, 'image', split)
        self.annotations_dir = os.path.join(root, 'annotation', split)
        self.images = list(sorted(os.listdir(self.images_dir)))
        self.masks = list(sorted(os.listdir(self.annotations_dir)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.annotations_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = process_label(mask, self.class_mapping)
        if self.transform_image is not None:
            image = self.transform_image(image)
        if self.transform_mask is not None:
            mask = self.transform_mask(mask)
        return image, mask

transform_image = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])

transform_mask = transforms.Compose([
    transforms.Resize((512, 512), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze().long())  # Ensure mask is LongTensor and 3D
])

data_root = 's3-data'

train_dataset = CustomDataset(root=data_root, split='train', class_mapping=class_mapping, transform_image=transform_image, transform_mask=transform_mask)
val_dataset = CustomDataset(root=data_root, split='val', class_mapping=class_mapping, transform_image=transform_image, transform_mask=transform_mask)
test_dataset = CustomDataset(root=data_root, split='test', class_mapping=class_mapping, transform_image=transform_image, transform_mask=transform_mask)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
model.classifier[4] = torch.nn.Conv2d(256, 16, kernel_size=(1, 1), stride=(1, 1))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def compute_iou(preds, masks, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = masks == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

num_epochs = 8
best_miou = 0.0
best_mious = []
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    model.eval()
    val_loss = 0.0
    all_ious = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            ious = compute_iou(preds, masks, num_classes=16)  # 16 classes including background
            all_ious.append(ious)

    mean_ious = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_ious)
    print(f"Validation Loss: {val_loss / len(val_loader)}, mIoU: {miou}")
    for idx, class_name in enumerate(METAINFO["classes"]):
        print(f"IoU for class {class_name}: {mean_ious[idx]}")

    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), 'best_deeplabv3_custom.pth')

print(f"Best mIoU: {best_miou}")
