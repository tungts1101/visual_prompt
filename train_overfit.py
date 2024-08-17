import torch
import torch.nn.functional as F
from dataloader import get_dataset, get_dataloader
from tqdm import tqdm
from omegaconf import OmegaConf
import timm
import util
import json
from pathlib import Path


def main():
    util.set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = OmegaConf.load('cifar100.yaml')
    print('Training model to overfit')
    config.dataset.percentage = 1
    train_dataset, val_dataset, num_classes = get_dataset(config)
    train_loader, val_loader = get_dataloader(config, train_dataset, val_dataset)
    
    model = timm.create_model('efficientnet_b0', pretrained=True, 
                              num_classes=num_classes)
    model = model.to(device)
    util.get_model_info(model)
    model_ckpt_path = Path('model_overfit.pth')
    
    if not Path.exists(model_ckpt_path):
        train_overfit(model, train_loader, device)
        
    # confirm that model is overfitting
    evaluate_model(model, val_loader, device)


def train_overfit(model, train_loader, device):
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    epoch = 0
    best_train_accuracy = 0
    no_improvement_epoch_count = 0
    
    while True:
        train_loss, train_accuracy = train(model, train_loader, optimizer, device, epoch)
        if train_accuracy > best_train_accuracy + 0.01:
            best_train_accuracy = train_accuracy
            no_improvement_epoch_count = 0
            torch.save(model.state_dict(), 'model_overfit.pth')
        else:
            no_improvement_epoch_count += 1
            if no_improvement_epoch_count == 10:
                break
        epoch += 1
        scheduler.step()


def train(model, train_loader, optimizer, device, epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'Epoch {epoch+1}')
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss=running_loss/(i+1), acc=correct/total)
    return running_loss, 100 * correct/total


def evaluate_model(model, val_loader, device):
    model.eval()
    top1 = util.AverageMeter('Acc@1')
    top5 = util.AverageMeter('Acc@5')
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            acc1, acc5 = util.accuracy(outputs, labels, topk=(1, 5))
            top1.update(acc1[0].item(), inputs.size(0))
            top5.update(acc5[0].item(), inputs.size(0))
            
        stats = dict(acc1=top1.avg, acc5=top5.avg)
        print(json.dumps(stats))


if __name__ == '__main__':
    main()