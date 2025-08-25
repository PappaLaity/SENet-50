import torch
from functions.functions import senet50, test
from models.dataset import CifarImageDataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader



if __name__ == "__main__":
    # load Data
    # Example Usage:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CifarImageDataset(csv_file='datasets/cifar-10/trainLabels.csv', image_folder='datasets/cifar-10/train', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print(dataset.__getitem__(3))
    # datasets/cifar-10
    # datasets/cifar-10/trainLabels.csv
    # datasets/cifar-10/train

    # print(f"Dataloader len {len(dataloader.dataset)}")
    # Binaire
    model_bin = senet50(num_classes=1)
    out_bin = model_bin(torch.randn(2, 3, 224, 224))
    print("Binaire:", out_bin.shape)  # torch.Size([2, 1])

    # Multi-classes (ex: Cifar 10)
    model_multi = senet50(num_classes=5)
    out_multi = model_multi(torch.randn(2, 3, 224, 224))
    print("Multi-classes:", out_multi.shape)  # torch.Size([2, 5])
