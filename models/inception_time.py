import torch
import torch.nn as nn

def conv(in_planes, out_planes, kernel_size=3, stride=1):
    "convolution with padding"
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

class InceptionBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_bottleneck=True):
        super(InceptionBlock1d, self).__init__()
        
        if use_bottleneck:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, stride=1, bias=False)
        else:
            self.bottleneck = nn.Identity()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(bottleneck_channels if use_bottleneck else in_channels, out_channels, kernel_size=k, padding=(k-1)//2, bias=False)
            for k in kernel_sizes
        ])
        
        self.pool_conv = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )
        
        self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))
        self.gelu = nn.GELU()

    def forward(self, x):
        x_bottleneck = self.bottleneck(x)
        x_convs = [conv(x_bottleneck) for conv in self.convs]
        x_pool = self.pool_conv(x)
        x_out = torch.cat(x_convs + [x_pool], dim=1)
        x_out = self.batch_norm(x_out)
        return self.gelu(x_out)

class Shortcut1d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Shortcut1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, residual):
        x_shortcut = self.conv(x)
        x_shortcut = self.batch_norm(x_shortcut)
        return nn.GELU()(x_shortcut + residual)

class InceptionNetwork(nn.Module):
    def __init__(self, num_blocks=6, in_channels=1, out_channels=32, bottleneck_channels=32, num_classes=1, residual=True):
        super(InceptionNetwork, self).__init__()
        
        self.blocks = nn.ModuleList()
        self.residual = residual
        
        for i in range(num_blocks):
            self.blocks.append(InceptionBlock1d(
                in_channels=in_channels if i == 0 else out_channels * 4, 
                out_channels=out_channels,
                bottleneck_channels=bottleneck_channels,
                use_bottleneck=True
            ))
        
        if residual:
            self.shortcuts = nn.ModuleList([
                Shortcut1d(in_channels if i == 0 else out_channels * 4, out_channels * 4)
                for i in range(num_blocks // 3)
            ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(out_channels * 4, num_classes)

    def forward(self, x):
        input_res = x
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            if self.residual and (i % 3 == 2):
                x = self.shortcuts[i // 3](input_res, x)
                input_res = x.clone()
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    
class InceptionNetworkWithDownsampling62(nn.Module):
    def __init__(self, num_blocks=6, in_channels=12, out_channels=32, bottleneck_channels=32, num_classes=1, residual=True):
        super(InceptionNetworkWithDownsampling62, self).__init__()
        
        self.inception_network = InceptionNetwork(
            num_blocks=num_blocks, 
            in_channels=out_channels, 
            out_channels=out_channels, 
            bottleneck_channels=bottleneck_channels, 
            num_classes=num_classes, 
            residual=residual
        )
        self.down_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.down_layers(x)
        return self.inception_network(x)
    
class InceptionNetworkWithDownsampling125(nn.Module):
    def __init__(self, num_blocks=6, in_channels=12, out_channels=32, bottleneck_channels=32, num_classes=1, residual=True):
        super(InceptionNetworkWithDownsampling125, self).__init__()
        self.inception_network = InceptionNetwork(
            num_blocks=num_blocks, 
            in_channels=out_channels, 
            out_channels=out_channels, 
            bottleneck_channels=bottleneck_channels, 
            num_classes=num_classes, 
            residual=residual
        )
        self.down_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.down_layers(x)
        return self.inception_network(x)

class InceptionNetworkWithDownsampling250(nn.Module):
    def __init__(self, num_blocks=6, in_channels=12, out_channels=32, bottleneck_channels=32, num_classes=1, residual=True):
        super(InceptionNetworkWithDownsampling250, self).__init__()
        self.inception_network = InceptionNetwork(
            num_blocks=num_blocks, 
            in_channels=out_channels, 
            out_channels=out_channels, 
            bottleneck_channels=bottleneck_channels, 
            num_classes=num_classes, 
            residual=residual
        )
        self.down_layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.down_layers(x)
        return self.inception_network(x)