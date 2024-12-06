from pointcept.models import build_model
import torch.nn as nn

class SparseConvModel(nn.Module):
    def __init__(self, 
                 type, in_channels, 
                 base_channels, channels, layers, stride):
        super(SparseConvModel, self).__init__()
        self.model = build_model(
            dict(
                type=type,
                in_channels=in_channels,
                num_classes=0, #no last layers
                base_channels=base_channels,
                channels=channels,
                layers=layers,
                strides=stride,
                cls_mode=False,
            )
        )
        self.output_dim = channels[-1]
    def forward(self, x,**kwargs):
        return self.model(x)
