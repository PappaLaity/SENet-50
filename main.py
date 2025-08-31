import pandas as pd
import torch
# import torch.nn as nn
from functions.functions import  se_resnet50, train_and_evaluate, train_and_evaluate_v2
# from models.dataset import CifarImageDataset
import torchvision
import wandb
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torchvision.models import resnet50, resnet101




if __name__ == "__main__":

    wandb.login(key="b1b7206839df8b716bebc62952a19f3a54f2f7b1")

    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    num_classes = 10 #100 for Cifar-100

    # Datasets For Cifar-10

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size

    train_subset, val_subset = random_split(trainset, [train_size, val_size])

    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128,
                                            shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(val_subset, batch_size=128,
                                         shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)
    
    # CIFAR 100
    # trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
    #                                     download=True, transform=transform)
    # train_size = int(0.8 * len(trainset))
    # val_size = len(trainset) - train_size

    # train_subset, val_subset = random_split(trainset, [train_size, val_size])

    # trainloader = torch.utils.data.DataLoader(train_subset, batch_size=128,
    #                                         shuffle=True, num_workers=2)

    # valloader = torch.utils.data.DataLoader(val_subset, batch_size=128,
    #                                      shuffle=False, num_workers=2)

    # testset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=128,
    #                                         shuffle=False, num_workers=2)

    results = {}
    resnet_50_model = resnet50(weights=None, num_classes=num_classes) 
    resnet_101_model = resnet101(weights=None, num_classes=num_classes)
    se_resnet_50_model = se_resnet50(num_classes)
    models = [resnet_50_model,resnet_101_model,se_resnet_50_model]
    # models = [resnet_50_model,resnet_101_model]
    models_name = ['ResNet-50','ResNet-101','SE-ResNet-50']
    # models_name = ['ResNet-50','ResNet-101']

    for idx,model in enumerate(models):
        model_name = models_name[idx]
        result = train_and_evaluate_v2(model,model_name,device,trainloader,valloader,testloader,wandb,"CIFAR-10",10)
        results[model_name] = result


    print(results)

    # --- 5. Affichage des résultats ---
    # print("\n" + "="*50)
    # print("             COMPARISON")
    # print("="*50)
    # for model_name, data in results.items():
    #     print(f"Model: {model_name}")
    #     print(f"  Test Accuracy: {data['Accuracy']:.2f}%")
    #     print(f"  Times: {data['Time']:.2f} seconds")
    #     print("-" * 30)
