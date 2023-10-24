import os
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2

data_path = os.getcwd()

labels = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))

# print(labels.head())

mpl.rc('font', size=15)
plt.figure(figsize=(7, 7))
label = ["Has cactus", "Hasnt cactus"]
plt.pie(labels['has_cactus'].value_counts(), labels=label, autopct="%.1f%%")

# plt.show()

######################
# Count train, test file
######################

num_train = len(os.listdir("train/"))
num_test = len(os.listdir("test/"))

print(f"Number of Train : {num_train}, Test : {num_test}")

mpl.rc('font', size=7)
plt.figure(figsize=(15, 6))
grid = gridspec.GridSpec(2, 6)

# last_has_cactus_img_name = labels[labels['has_cactus'] == 1]['id'][-12:]
last_has_cactus_img_name = labels[labels['has_cactus'] == 0]['id'][-12:]

for idx, img_name in enumerate(last_has_cactus_img_name):
    img_path = 'train/' + img_name
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(grid[idx])
    ax.imshow(image)
plt.show()
