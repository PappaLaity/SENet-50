import pandas as pd
import torch
import torch.nn as nn
from functions.functions import evaluate, senet50, train
from models.dataset import CifarImageDataset
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader,random_split



if __name__ == "__main__":

    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    # print(df['label'].value_counts())  Balanced Data 5000 for each Category
    # load Data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # resize it to 224*224
    dataset = CifarImageDataset(csv_file='datasets/cifar-10/trainLabels.csv', image_folder='datasets/cifar-10/train', transform=transform)

    # Taille des splits
    train_size = int(0.8 * len(dataset))  # 80% pour train
    val_size = len(dataset) - train_size  # 20% pour validation

    # Split aléatoire
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


    # Multi-classes (ex: Cifar 10)
    model_multi = senet50(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_multi.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    print("Model Senet 50")
    train(model_multi,device,50,train_loader,criterion,optimizer,scheduler)
    print("Model Senet 50 Evaluation")
    evaluate(model_multi,val_loader,device) 

    print("Model Resnet 50")
    model = models.resnet50(weights=None)
    num_classes = 10
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # scheduler=None
    train(model,device,50,train_loader,criterion,optimizer,scheduler)
    print("Model Resnet 50 Evaluation")
    evaluate(model,val_loader,device) 

    print("Model Resnet 101")

    model101 = models.resnet101(weights=None)
    num_classes = 10
    model101.fc = nn.Linear(model.fc.in_features, num_classes)
    optimizer = torch.optim.Adam(model101.parameters(),lr=0.1,weight_decay=5e-4)

    optimizer = torch.optim.SGD(model101.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # scheduler = None

    train(model101,device,50,train_loader,criterion,optimizer,scheduler)
    print("Model Resnet 101 Evaluation")
    evaluate(model101,val_loader,device) 