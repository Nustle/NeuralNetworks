import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import timm
from tqdm import tqdm

from PIL import Image
import os
from torchvision import transforms
import random
import torch.optim as optim
from sklearn.metrics import f1_score

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(42)

labels_df = pd.read_csv('labels.csv')
print(labels_df.shape)


# 1. Создаём датасет
'''
Принимает метки классов, путь к директории с изображениями и аугментации.
'''
class ImageDataset(Dataset):
    def __init__(self, labels_df, img_dir, transform=None):
        self.labels_df = labels_df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = str(self.labels_df.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label}
        return sample


# 2. Аугментации
'''
1) Resize - фиксированный размер ускоряет обучение и можно использовать batch_size > 1.
2) RandomHorizontalFlip - горизонтальное отражение.
3) RandomAffine - сдвигает, масштабирует и поворачивает изображение.
4) ColorJitter - изменяет яркость и контраст изображения, а также 
   измененяет цвета изображения в пространстве HSV.
5) Статистики датасета ImageNet.
'''
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.8, 1.2),
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.3,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_df, val_df = train_test_split(
    labels_df,
    test_size=0.2,
    random_state=42,
    stratify=labels_df.iloc[:, 1]
)

train_dataset = ImageDataset(
    labels_df=train_df.reset_index(drop=True),
    img_dir='data',
    transform=train_transforms
)

val_dataset = ImageDataset(
    labels_df=val_df.reset_index(drop=True),
    img_dir='data',
    transform=val_transforms
)

batch_size = 32
num_workers = 2 if device == "cuda" else 0
pin_memory = True if device == "cuda" else False

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

# 3. Классификатор
'''
В качестве основной модели берётся MobileNet, в которой
изменяется классификационная голова.
'''
class ModelClassifier(nn.Module):
    def __init__(self, num_classes=50, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrained
        )

        num_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


model = ModelClassifier(num_classes=50).to(device)

# замораживаются все веса, кроме весов из классификационной головы.
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader)
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        current_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        pbar.set_postfix({
            'loss': f'{running_loss / len(pbar):.4f}',
            'f1': f'{current_f1:.4f}'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_f1


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader)
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

            pbar.set_postfix({
                'loss': f'{running_loss / len(pbar):.4f}',
                'f1': f'{current_f1:.4f}'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return epoch_loss, epoch_f1


# 4. Обучение модели
'''
Используется finetuning подход, а точнее механизм плавной разморозки.
Первые 7 эпох заморожены все веса, кроме весов из классифкационной головы.
В следующих 8 эпохах модель обучает веса 6-ого backbone слоя.
Для последних 25 эпох размораживаются и обучаются все веса модели.

Для каждого этапа задаётся свой lr. 
'''
def train_model(model, train_loader, val_loader,
                epochs_per_stage=[7, 8, 25],
                learning_rates=[1e-3, 1e-4, 1e-5],
                device='cuda'):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {
        'train_loss': [],
        'train_f1': [],
        'val_loss': [],
        'val_f1': []
    }

    stages = [
        ("Stage 1: Only Classifier", ['backbone.classifier']),
        ("Stage 2: Classifier and Last Blocks", ['backbone.blocks.6', 'backbone.classifier']),
        ("Stage 3: Full Model", None)
    ]

    best_val_f1 = 0.0
    current_epoch = 0

    for stage_idx, (stage_name, layers_to_unfreeze) in enumerate(stages):
        if layers_to_unfreeze is None:
            for param in model.parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = False

            for layer_name in layers_to_unfreeze:
                layer = model
                for attr in layer_name.split('.'):
                    layer = getattr(layer, attr)
                for param in layer.parameters():
                    param.requires_grad = True

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params:,}")

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rates[stage_idx]
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.5
        )

        epochs = epochs_per_stage[stage_idx]

        for epoch in range(epochs):
            current_epoch += 1
            print(f"\nEpoch {current_epoch} (Stage {stage_idx + 1})")

            train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_f1 = validate(model, val_loader, criterion, device)

            history['train_loss'].append(train_loss)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)

            scheduler.step(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'model.pth')
                print(f"Best model saved! Val F1: {val_f1:.4f}")

    return model, history


model, history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device
)
