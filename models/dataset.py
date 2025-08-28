import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os

class CifarImageDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
        self.labels = self.annotations['label']
        self.label2idx = {label: idx for idx, label in enumerate(self.annotations.iloc[:,1].unique())}


    def __len__(self):
        return len(self.annotations)
    
    def __label2idx__(self):
        return self.label2idx

    def __getitem__(self, index):
        img_name = str(self.annotations.iloc[index, 0])+".png" # Assuming image name is in the first column
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB") # Load image and convert to RGB
        target_str = self.annotations.iloc[index, 1] # Assuming target is in the second column
        target = self.label2idx[target_str] 

        if self.transform:
            image = self.transform(image)

        return image, target

