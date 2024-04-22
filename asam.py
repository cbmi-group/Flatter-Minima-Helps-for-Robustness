import torch
from collections import defaultdict
from copy import deepcopy
import torch.nn as nn


def layer_sharpness(args, dataloader, model, criterion, epsilon=0.1):

    model.eval()
    layer_sharpness_dict = {} 
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # print(name)
            # For WideResNet
            if "sub" in name:
                continue
            layer_sharpness_dict[name+".weight"] = 1e10

    for layer_name, _ in model.named_parameters():
        if "weight" in layer_name and layer_name in layer_sharpness_dict.keys():
            # print(layer_name)
            cloned_model = deepcopy(model)
            # set requires_grad sign for each layer
            for name, param in cloned_model.named_parameters():
                # print(name)
                if name == layer_name:
                    # print(name)
                    param.requires_grad = True
                    init_param = param.detach().clone()
                else:
                    param.requires_grad = False
        
            optimizer = torch.optim.SGD(cloned_model.parameters(), lr=1)

            max_loss = 0.0
            min_acc = 0
    
            for epoch in range(2):
                # Gradient ascent
                for inputs, targets in dataloader:
                    optimizer.zero_grad()
                    outputs = cloned_model(inputs)
                    loss = -1 * criterion(outputs, targets) 
                    loss.backward()
                    optimizer.step()
                sd = cloned_model.state_dict()
                diff = sd[layer_name] - init_param
                times = torch.linalg.norm(diff)/torch.linalg.norm(init_param)
                # print(times)
                if times > epsilon:
                    diff = diff / times * epsilon
                    sd[layer_name] = deepcopy(init_param + diff)
                    cloned_model.load_state_dict(sd)

                with torch.no_grad():
                    total = 0
                    total_loss = 0.0
                    correct = 0
                    for inputs, targets in dataloader:
                        outputs = cloned_model(inputs)
                        total += targets.shape[0]
                        total_loss += criterion(outputs, targets).item() * targets.shape[0]
                        _, predicted = outputs.max(1)
                        correct += predicted.eq(targets).sum().item()  
                    
                    total_loss /= total
                    correct /= total
                    # print("After {}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format(epoch, total_loss, correct*100))
                if total_loss > max_loss:
                    max_loss = total_loss
                    min_acc = correct
            
            layer_sharpness_dict[layer_name[:-len(".weight")]] = max_loss
            args.logger.info("{:35}, Robust Loss: {:10.2f}, Robust Acc: {:10.2f}".format(layer_name[:-len(".weight")], max_loss, min_acc*100))

    for (k, v) in layer_sharpness_dict.items():
        args.logger.info("{:35}, Robust Loss: {:10.2f}".format(k, v))
    
    return layer_sharpness_dict


class ASAM:
    def __init__(self, optimizer, model, rho=0.5, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.rho / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class Layer_ASAM:
    def __init__(self, optimizer, model, layer_rho=None, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.layer_rho = layer_rho
        self.eta = eta
        self.state = defaultdict(dict)

    def updat_layer_rho(self, layer_rho):
        self.layer_rho = layer_rho

    @torch.no_grad()
    def ascent_step(self):
        wgrads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if t_w is None:
                t_w = torch.clone(p).detach()
                self.state[p]["eps"] = t_w
            if 'weight' in n:
                t_w[...] = p[...]
                t_w.abs_().add_(self.eta)
                p.grad.mul_(t_w)
            wgrads.append(torch.norm(p.grad, p=2))
        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            t_w = self.state[p].get("eps")
            if 'weight' in n:
                p.grad.mul_(t_w)
            eps = t_w
            eps[...] = p.grad[...]
            eps.mul_(self.layer_rho[n] / wgrad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def descent_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"])
        self.optimizer.step()
        self.optimizer.zero_grad()


class SAM(ASAM):
    @torch.no_grad()
    def ascent_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            p.add_(eps)
        self.optimizer.zero_grad()