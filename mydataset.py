import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

class MyDataset(torch.utils.data.Dataset):
    
    def __init__(self, df, image_dir):
        
        super().__init__()
        
        self.image_ids = df["image_id"].unique()
        self.df = df
        self.image_dir = image_dir
        
    def __getitem__(self, index):
 
        transform = transforms.Compose([
                                        transforms.ToTensor()
        ])
 
        # 入力画像の読み込み
        image_id = self.image_ids[index]
        image = Image.open(f"{self.image_dir}/{image_id}.jpg")
        image = transform(image)
        
        # アノテーションデータの読み込み
        records = self.df[self.df["image_id"] == image_id]
        boxes = torch.tensor(records[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32), dtype=torch.float32)
        
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)
        
        labels = torch.tensor(records["class"].values.astype(np.int64), dtype=torch.int64)
        
        iscrowd = torch.zeros((records.shape[0], ), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"]= labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowd"] = iscrowd


        return image, labels[0], image_id
    
    def __len__(self):
        return self.image_ids.shape[0]

if __name__=="__main__":
    pass
