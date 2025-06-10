import torch
import torch.nn as nn

# 多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[64, 32, 16]):
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.Sigmoid(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
