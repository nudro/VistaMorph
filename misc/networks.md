# Network Architecture Options for `self.fc_loc`

This document outlines various architectural options for the `self.fc_loc` network while maintaining the required input dimension of `1*17*768` and output dimension of `3*2` with a final `Sigmoid` activation.

## Common Requirements
- Input dimension: `1*17*768`
- Output dimension: `3*2`
- Final activation: `nn.Sigmoid()`
- Intermediate layer size: `256`

## 1. Simple Two-Layer Network
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 2. Deeper Network with Dropout
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 3. Network with Batch Normalization
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024, bias=True),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Linear(1024, 256, bias=True),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Linear(256, 3*2, bias=True),
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
            nn.ReLU(),
            nn.Linear(in_features, in_features)
        )
    
    def forward(self, x):
        return x + self.block(x)

self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.ReLU(),
    ResidualBlock(1024),
    nn.Linear(1024, 256),
    nn.ReLU(),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 5. Network with LeakyReLU and Layer Normalization
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.LayerNorm(1024),
    nn.LeakyReLU(0.2),
    nn.Linear(1024, 512),
    nn.LayerNorm(512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 6. Network with GELU Activation
```python
self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024),
    nn.GELU(),
    nn.Linear(1024, 512),
    nn.GELU(),
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Linear(256, 3*2),
    nn.Sigmoid()
)
```

## 7. Network with Skip Connections
```python
class SkipConnection(nn.Module):
    def __init__(self, in_features, out_features):
        super(SkipConnection, self).__init__()
        self.skip = nn.Linear(in_features, out_features, bias=True)
        self.main = nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.ReLU(),
            nn.Linear(out_features, out_features, bias=True)
        )
    
    def forward(self, x):
        return self.skip(x) + self.main(x)

self.fc_loc = nn.Sequential(
    nn.Linear(1*17*768, 1024, bias=True),
    nn.ReLU(),
    SkipConnection(1024, 256),
    nn.Linear(256, 3*2, bias=True),
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

## Note on Bias Parameters
For options 3 and 7, we explicitly set `bias=True` in the Linear layers to ensure proper initialization and avoid the "ReLU object has no attribute 'bias'" error. This is particularly important when using BatchNorm1d and Skip Connections. 