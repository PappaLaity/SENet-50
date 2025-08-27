import torch
from models.senet50_v2 import SENet50


def senet50(num_classes=1, reduction=16):
    return SENet50(num_classes=num_classes, reduction=reduction)


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

def test():
    pass