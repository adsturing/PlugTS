import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, dim, hidden_size=100):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_size)
        self.activate = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size+1, 1, bias=False)
    def forward(self, x):
        h = self.activate(self.fc1(x))
        h = torch.cat((h, torch.Tensor(np.ones((h.size()[0],1))).cuda()),1)
        return self.fc2(h), h
        
class PlugTS:
    def __init__(self, dim, lamdba=1, nu=1, hidden=100):
        self.func = Network(dim, hidden_size=hidden).cuda()
        self.context_list = None
        self.len = 0
        self.reward = None
        self.lamdba = lamdba
        self.total_param = sum(p.numel() for name, p in self.func.named_parameters() if p.requires_grad and name[:3] == 'fc2')
        self.U = lamdba * torch.ones((self.total_param,)).cuda()
        self.nu = nu
        self.loss_func = nn.MSELoss()

    def select(self, context):
        tensor = torch.from_numpy(context).float().cuda()
        with torch.no_grad():
            mu, grad = self.func(tensor)
        sigma = torch.sqrt(torch.sum(self.lamdba * self.nu * grad * grad / self.U, dim=1))
        sample_r = torch.normal(mu.view(-1), sigma.view(-1))
        arm = torch.argmax(sample_r)
        self.U += grad[arm] * grad[arm]
        return arm
    
    def train(self, context, reward):
        self.len += 1
        optimizer = optim.SGD(self.func.parameters(), lr=1e-2, weight_decay=self.lamdba / self.len)
        if self.context_list is None:
            self.context_list = torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)
            self.reward = torch.tensor([reward], device='cuda', dtype=torch.float32)
        else:
            self.context_list = torch.cat((self.context_list, torch.from_numpy(context.reshape(1, -1)).to(device='cuda', dtype=torch.float32)))
            self.reward = torch.cat((self.reward, torch.tensor([reward], device='cuda', dtype=torch.float32)))

        for _ in range(100):
            optimizer.zero_grad()
            pred, _ = self.func(self.context_list)
            loss = self.loss_func(pred.view(-1), self.reward)
            loss.backward()
            optimizer.step()
        return loss
