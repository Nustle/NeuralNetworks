import torch
import torch.nn as nn
import timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

if len(sys.argv) > 1:
    path_dir = sys.argv[1]
else:
    path_dir = input("Путь к каталогу path: ")

test_dir = f"{path_dir}/test"
model_path = "model.pth"
output_csv = "label_test.csv"


class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(img_dir)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        sample = {"img": image, "img_name": img_name}
        return sample


test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

batch_size = 32
num_workers = 2 if device == "cuda" else 0

test_dataset = TestDataset(test_dir, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


class ModelClassifier(nn.Module):
    def __init__(self, num_classes=50, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            'mobilenetv3_large_100',
            pretrained=pretrained
        )

        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def predict():
    model = ModelClassifier(num_classes=50, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    filenames = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            images = batch['img'].to(device)
            img_names = batch['img_name']

            outputs = model(images)
            _, predicted = outputs.max(1)

            predictions.extend(predicted.cpu().numpy())
            filenames.extend(img_names)

    results_df = pd.DataFrame({
        'filenames': filenames,
        'label': predictions
    })

    results_df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


predict()
