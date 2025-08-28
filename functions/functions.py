import torch
import torch.nn as nn
import time
from models.senet50 import Bottleneck, SEResNet
from models.senet50_v2 import SENet50


def senet50(num_classes=1, reduction=16):
    return SENet50(num_classes=num_classes, reduction=reduction)

def se_resnet50(num_classes=10):
    return SEResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def train(model,device,epochs,dataloader,criterion,optimizer,scheduler):
    # Train Mode
    model.train()

    # Use Gpu or CPU depending of what resource we have
    model.to(device)
    
    # Exponential moving Average loss
    ema_loss = None

    # For each epochs
    for epoch in range(epochs):
        # For Each Batch in dataloader
        for batch_idx, (data,target) in enumerate(dataloader):
            
            # Convert it to device
            data = data.to(device, dtype=torch.float32)
            target = target.to(device)
            
            # Train Data Forward Pass
            output = model(data)
            
            # Calculate loss
            loss = criterion(output.to(device),target)

            # Backward Pass

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ema_loss == None:
                ema_loss = loss.item()
            else:
                ema_loss+= (loss.item() - ema_loss)*0.01
            
            if scheduler:
                scheduler.step()
            
        print("Train Epoch {} \t loss: {:.6f}".format(epoch,ema_loss))


def evaluate(model,dataloader,device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data,target in dataloader:
            
            data = data.to(device, dtype=torch.float32)
            # target = target.to(device)

            output = model(data)

            pred = output.argmax(dim =1 ,keepdim = True)

            correct += pred.cpu().eq(target.view_as(pred)).sum().item()

    percent = (correct*100)/len(dataloader.dataset)

    print(f"Accuracy: {correct} / {len(dataloader.dataset)} ({percent:.2f}%)")

    return percent



# --- 3. Fonction d'entraînement et d'évaluation ---
def train_and_evaluate(model,model_name,device,trainloader,testloader,num_epochs=100):

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    print(f"\n--- Début de l'entraînement pour {model_name} ---")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        scheduler.step()

        # Évaluation sur le jeu de test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

    end_time = time.time()
    training_time = end_time - start_time
    print(f"--- Entraînement terminé pour {model_name} en {training_time:.2f} secondes. ---")
    return accuracy, training_time