# Network Architecture Options for `self.fc_loc`

This document outlines various architectural options for the `self.fc_loc` network while maintaining the required input dimension of `1*17*768` and output dimension of `3*2` with a final `Sigmoid` activation.

## Common Requirements
- Input dimension: `1*17*768`
- Output dimension: `3*2`
- Final activation: `nn.Sigmoid()`
- Intermediate layer size: `256`
- Note: All architectures ensure that `self.fc_loc[2]` accesses a Linear layer for proper bias initialization

## 1. Simple Two-Layer Network
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 2. Deeper Network with Dropout
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 512),
    nn.ReLU(True),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(True),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 3. Network with Batch Normalization
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.BatchNorm1d(1024),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    nn.BatchNorm1d(256),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 4. Residual-like Connections
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(True),
            nn.Linear(in_features, in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    ResidualBlock(256),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 5. Network with LeakyReLU and Layer Normalization
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.LayerNorm(1024),
    nn.Linear(1024, 512),
    nn.ReLU(True),
    nn.LayerNorm(512),
    nn.Linear(512, 256),
    nn.ReLU(True),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 6. Network with GELU Activation
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.GELU(),
    nn.Linear(1024, 512),
    nn.ReLU(True),
    nn.GELU(),
    nn.Linear(512, 256),
    nn.ReLU(True),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 7. Network with Skip Connections
```python
class SkipConnection(nn.Module):
    def __init__(self, in_features, out_features):
        super(SkipConnection, self).__init__()
        self.skip = nn.Linear(in_features, out_features)
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(True),
            nn.Linear(out_features, out_features)
        )
    
    def forward(self, x):
        return self.skip(x) + self.main(x)

self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(True),
    nn.Linear(1024, 256),
    nn.ReLU(True),
    SkipConnection(256, 256),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## Architecture Selection Guide

Choose an architecture based on your specific needs:

- **Better Regularization**: Use option 2 with dropout
- **Better Normalization**: Use option 3 with batch normalization
- **Prevent Vanishing Gradients**: Use option 4 with residual connections
- **Modern Activation**: Use option 6 with GELU
- **Simpler Architecture**: Use option 1
- **Skip Connections**: Use option 7 for better gradient flow
- **Layer Normalization**: Use option 5 for more stable training

Each architecture maintains the required input and output dimensions while offering different characteristics in terms of:
- Number of layers
- Activation functions (ReLU, LeakyReLU, GELU)
- Normalization techniques (BatchNorm, LayerNorm)
- Regularization (Dropout)
- Connection patterns (Residual, Skip connections)

## Note on Layer Ordering
All architectures are structured to ensure that:
1. Linear layers are properly positioned for bias access
2. Activation functions (ReLU, GELU) come after Linear layers
3. Normalization layers (BatchNorm, LayerNorm) come after activations
4. The final layer is always a Linear layer followed by Sigmoid
This structure ensures compatibility with bias initialization code like `self.fc_loc[2].bias.data.zero_()`. 