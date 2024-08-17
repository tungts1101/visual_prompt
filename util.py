import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_model_info(model):
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}\
          \nNumber of trainable parameters: {num_trainable_params}\
          \nPercetage of trainable parameters: {num_trainable_params/num_params*100:.2f}%')
    

def info_nce_loss(query, positive, negatives, temperature=0.07):
    """
    Compute InfoNCE Loss

    Args:
        query (Tensor): (batch_size, dim)
        positive (Tensor): (batch_size, dim)
        negatives (_type_): (batch_size, dim') or (batch_size, num_negatives, dim')
        temperature (float, optional): Temperature for scaling the logits. Defaults to 0.07.
    """
    batch_size, dim = query.shape
    if negatives.dim() == 2:
        negatives = negatives.unsqueeze(1) # (batch_size, 1, dim)
    num_negatives = negatives.shape[1]
    
    # ensure negatives have the same dimensionality as the query and positive
    negative_dim = negatives.shape[-1]
    if negative_dim < dim:
        padding = torch.zeros((batch_size, num_negatives, dim - negative_dim), device=negatives.device)
        negatives = torch.cat([negatives, padding], dim=-1)
    elif negative_dim > dim:
        negatives = negatives[:, :, :dim]
    
    # compute similarity scores as dot products between query and keys
    query = query.unsqueeze(2) # (batch_size, dim, 1)
    # positive dot products (batch_size, 1)
    positive_dot = torch.bmm(query.transpose(1, 2), positive.unsqueeze(2))
    positive_dot = positive_dot.flatten().reshape(batch_size, 1)
    # negative dot products (batch_size, num_negatives)
    negative_dots = torch.bmm(query.transpose(1, 2), negatives.transpose(1, 2))
    negative_dots = negative_dots.flatten().reshape(batch_size, num_negatives)
    logits = torch.cat([positive_dot, negative_dots], dim=1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss
    

def get_samples_by_shift(x, max_shift=30, num_samples=64):
    """
    Generate num_samples samples by shifting the input tensor x.

    Args:
        x (Tensor): Input tensor of shape (batch_size, channels, height, width).
        max_shift (int, optional): Maximum shift value. Defaults to 30.
        num_samples (int, optional): Number of samples to generate. Defaults to 64.

    Returns:
        Tensor: Output tensor of shape (batch_size*num_samples, channels, height, width).
    """
    batch_size, *shape = x.shape
    samples = torch.empty((batch_size*num_samples, *shape), device=x.device)
    for i in range(batch_size):
        for j in range(num_samples):
            shift_x, shift_y = torch.randint(-max_shift, max_shift+1, (2,))
            sample = TF.affine(x[i].unsqueeze(0), angle=0, translate=(shift_x, shift_y), scale=1, shear=0)
            samples[i*num_samples+j] = sample
    return samples


def get_features(model, inputs, layer_names):
    """
    Extract features from specific layers of a model.

    Args:
    - model (nn.Module): The neural network model.
    - inputs (torch.Tensor): The input tensor to the model.
    - layer_names (list): A list of strings representing the layer names to extract features from.

    Returns:
    - features_dict (dict): A dictionary where keys are layer names and values are the corresponding extracted features.
    """
    features_dict = {}
    hooks = []

    # Define the hook function to save the output of each layer
    def hook_fn(name):
        def fn(module, input, output):
            features_dict[name] = output
        return fn

    # Register forward hooks on the layers specified in layer_names
    for name, layer in model.named_modules():
        if name in layer_names:
            hook = layer.register_forward_hook(hook_fn(name))
            hooks.append(hook)

    # Perform a forward pass to trigger the hooks
    with torch.no_grad():
        _ = model(inputs)

    # Remove the hooks after the forward pass
    for hook in hooks:
        hook.remove()

    return features_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res