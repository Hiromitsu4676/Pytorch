import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
from mydataset import MyDataset
from torchvision import transforms
from xml2list import Xml2List
from neuralnetwork import NeuralNetwork
import optuna


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y, z) in enumerate(dataloader):
        # X, y = X.to(device), y.to(device)

        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 5 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y ,z in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss


def objective(trial):
    #学習率
    lr = trial.suggest_uniform('lr', 1e-4, 1e-1)

    #エポック数
    epochs = trial.suggest_discrete_uniform('epochs', 3, 5, 1)

    optimizer = torch.optim.SGD(model.parameters(),lr)
    print('学習率',lr)
    print('エポック数',epochs)

    for step in range(int(epochs)):
        print('step:',step,'epoch:',epochs)
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss = test(test_dataloader, model)

    return test_loss


if __name__=="__main__":
    xml_paths = glob.glob("./Pytorch/privateDataset/xml_resize/*.xml")
    classes = ["saito", "ohnaka","doi",'sugai','suzuki']
    
    transform_anno = Xml2List(classes)

    df = pd.DataFrame(columns=["image_id", "width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
 
    for path in xml_paths:
        image_id = path.split("/")[-1].split(".")[0]
        bboxs = transform_anno(path)
        
        for bbox in bboxs:
            tmp = pd.Series(bbox, index=["width", "height", "xmin", "ymin", "xmax", "ymax", "class"])
            tmp["image_id"] = image_id
            df = df.append(tmp, ignore_index=True)
    
    df = df.sort_values(by="image_id", ascending=True)

    image_dir = "./Pytorch/privateDataset/img_resize"
    dataset = MyDataset(df, image_dir)

    batch_size = 1
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    test_dataloader = DataLoader(dataset, batch_size=batch_size)

    # 訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # modelを定義します
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()

    TRIAL_SIZE = 5
    study = optuna.create_study()
    study.optimize(objective, n_trials=TRIAL_SIZE)

    #最適化したハイパーパラメータの結果
    study.best_params





