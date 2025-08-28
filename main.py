import pandas as pd
import torch
# import torch.nn as nn
from functions.functions import  se_resnet50, train_and_evaluate
# from models.dataset import CifarImageDataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet101




if __name__ == "__main__":

    device  = "mps" if torch.backends.mps.is_available() else "cpu"
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    num_classes = 10 #100 for Cifar-100

    # Datasets For Cifar-10

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                            shuffle=False, num_workers=2)
    
    # CIFAR 100
    # trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
    #                                     download=True, transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
    #                                         shuffle=True, num_workers=2)

    # testset = torchvision.datasets.CIFAR100(root='./data', train=False,
    #                                     download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=128,
    #                                         shuffle=False, num_workers=2)

    results = {}


    resnet_50_model = resnet50(weights=None, num_classes=10) 
    resnet_50_accuracy, resnet_50_time = train_and_evaluate(resnet_50_model,'ResNet-50',device,trainloader,testloader)
    results['ResNet-50'] = {'Accuracy': resnet_50_accuracy, 'Time': resnet_50_time}

    # Modèle 3: ResNet-101
    resnet_101_model = resnet101(weights=None, num_classes=10)
    resnet_101_accuracy, resnet_101_time = train_and_evaluate(resnet_101_model,'ResNet-101',device,trainloader,testloader)
    results['ResNet-101'] = {'Accuracy': resnet_101_accuracy, 'Time': resnet_101_time}

    # Modèle 1: SE-ResNet-50
    se_resnet_50_model = se_resnet50(num_classes)
    se_accuracy, se_time = train_and_evaluate(se_resnet_50_model,'SE-ResNet-50',device,trainloader,testloader)
    results['SE-ResNet-50'] = {'Accuracy': se_accuracy, 'Time': se_time}

    # --- 5. Affichage des résultats ---
    print("\n" + "="*50)
    print("             RÉSUMÉ DE LA COMPARAISON")
    print("="*50)
    for model_name, data in results.items():
        print(f"Modèle: {model_name}")
        print(f"  Précision finale: {data['Accuracy']:.2f}%")
        print(f"  Temps d'entraînement: {data['Time']:.2f} secondes")
        print("-" * 30)
