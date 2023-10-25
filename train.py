import torch
import random
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split

import cv2
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

import torch.nn as nn

from sklearn.metrics import roc_auc_score

seed = 50
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True   # 확정적 연산 사용
torch.backends.cudnn.benchmark = False      # 벤치마크 기능 해제
torch.backends.cudnn.enabled = False        # cudnn 사용 해제

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_path = os.getcwd()

labels = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

train, valid = train_test_split(labels,
                                test_size=0.1,
                                stratify=labels['has_cactus'],
                                random_state=50)

print(f"train counting : {len(train)}")
print(f"valid counting : {len(valid)}")


class ImageDataset(Dataset):
    def __init__(self, df, img_dir='./', transform=None):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = self.img_dir + img_id
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[idx, 1]

        if self.transform is not None:
            image = self.transform(image)
        return image, label


transform = transforms.ToTensor()

dataset_train = ImageDataset(df=train, img_dir='train/', transform=transform)
dataset_valid = ImageDataset(df=valid, img_dir='train/', transform=transform)


def seed_worker(worker_id):
    '''
    DataLoader multi-processing
    '''
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

loader_train = DataLoader(dataset=dataset_train, batch_size=32,
                          shuffle=True)
                        #   worker_init_fn=seed_worker,
                        #   generator=g, num_workers=2)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=32,
                          shuffle=False)
                        #   worker_init_fn=seed_worker,
                        #   generator=g, num_workers=2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                              out_channels=32,
                                              kernel_size=3, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=32,
                                              out_channels=64,
                                              kernel_size=3, padding=2),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x


model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


epochs = 10

for epoch in range(epochs):
    epoch_loss = 0

    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch : [{epoch+1}/{epochs}] - loss : {epoch_loss/len(loader_train):.4f} ")

true_list = []
preds_list = []

model.eval()

with torch.no_grad():
    for images, labels in loader_valid:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        preds = torch.softmax(output.cpu(), dim=1)[:, 1]
        true = labels.cpu()

        preds_list.extend(preds)
        true_list.extend(true)

print(f"Validataion ROC AUC : {roc_auc_score(true_list, preds_list):.4f}")

dataset_test = ImageDataset(df=submission, img_dir='test/', transform=transform)
loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)

model.eval()

preds = []

with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)

        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        preds.extend(preds_part)

submission['has_cactus'] = preds
submission.to_csv('submission.csv', index=False)

print(f"[INFO] Make submission file.")
