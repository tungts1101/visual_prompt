import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from dataloader import get_dataset, get_dataloader
from omegaconf import OmegaConf
from tqdm import tqdm
import util
import json


class PromptModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt = nn.Parameter(torch.ones(1, 3, 224, 224))

    def forward(self, x):
        prompt = torch.cat(x.size(0) * [self.prompt])
        return x + prompt


class LinearProbeModel(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        features = self.model(x)
        return features, self.classifier(features)


def main():
    util.set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load('cifar100.yaml')
    config.dataset.percentage = 100
    train_dataset, val_dataset, num_classes = get_dataset(config)
    train_loader, val_loader = get_dataloader(config, train_dataset, val_dataset)
    
    pretrained_model = timm.create_model('resnet50', pretrained=True, num_classes=0)
    model = LinearProbeModel(pretrained_model, num_classes)
    model.to(device)
    util.get_model_info(model)
    
    negative_model = timm.create_model('efficientnet_b0', pretrained=True, 
                                       num_classes=num_classes)
    negative_model.to(device)
    negative_model.load_state_dict(torch.load('model_overfit.pth', weights_only=False))
    negative_model.requires_grad_(False)
    
    prompt_model = PromptModel()
    prompt_model.to(device)
    
    optimizer = torch.optim.SGD(list(model.classifier.parameters()) + list(prompt_model.parameters()), 
                                lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_acc = 0.0
    for epoch in range(300):
        results = train_prompt(model, prompt_model, negative_model, train_loader, optimizer, device, epoch)
        if results['vp_acc'] > best_acc:
            best_acc = results['vp_acc']
            torch.save(model.state_dict(), 'model_best.pth')
            torch.save(prompt_model.state_dict(), 'prompt_best.pth')
            
        scheduler.step()
        evaluate_model(model, prompt_model, val_loader, device)

def train_prompt(model, prompt_model, negative_model, train_loader, optimizer, device, epoch):
    cre_running_loss = 0.0
    nce_running_loss = 0.0
    lp_correct = 0
    vp_correct = 0
    total = 0
    
    pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader), desc=f'Epoch {epoch+1}')
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        prompts = prompt_model(inputs)
        
        query, lp_outputs = model(inputs)
        positive, vp_outputs = model(prompts)
        # negatives = util.get_features(negative_model, inputs, ['global_pool'])['global_pool']
        samples = util.get_samples_by_shift(inputs, num_samples=8)
        negatives = util.get_features(negative_model, samples, ['global_pool'])['global_pool']
        negatives = negatives.view(inputs.size(0), negatives.size(0) // inputs.size(0), -1)
        
        cre_loss = F.cross_entropy(vp_outputs, labels)
        nce_loss = util.info_nce_loss(query, positive, negatives)
        loss = cre_loss + nce_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cre_running_loss += cre_loss.item()
        nce_running_loss += nce_loss.item()
        total += labels.size(0)
        _, lb_predicted = lp_outputs.max(1)
        _, vp_predicted = vp_outputs.max(1)
        
        lp_correct += lb_predicted.eq(labels).sum().item()
        vp_correct += vp_predicted.eq(labels).sum().item()
        
        pbar.set_postfix(loss_cre=cre_running_loss/(i+1), loss_nce=nce_running_loss/(i+1),
                         acc_lp=lp_correct/total, acc_vp=vp_correct/total)
    return {
        'cre_loss': cre_running_loss, 'nce_loss': nce_running_loss,
        'lp_acc': 100 * lp_correct/total, 'vp_acc': 100 * vp_correct/total
    }

def evaluate_model(model, prompt_model, val_loader, device):
    model.eval()
    prompt_model.eval()
    lp_top1 = util.AverageMeter('Acc@1')
    vp_top1 = util.AverageMeter('Acc@1')
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            prompts = prompt_model(inputs)
            
            _, lp_outputs = model(inputs)
            _, vp_outputs = model(prompts)
            
            lp_acc1 = util.accuracy(lp_outputs, labels, topk=(1,))
            lp_top1.update(lp_acc1[0].item(), inputs.size(0))
            vp_acc1 = util.accuracy(vp_outputs, labels, topk=(1,))
            vp_top1.update(vp_acc1[0].item(), inputs.size(0))
            
        stats = dict(lp_acc=lp_top1.avg, vp_acc=vp_top1.avg)
        print(json.dumps(stats))


if __name__ == '__main__':
    main()