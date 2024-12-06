import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from PIL import Image
import cv2
import einops
from scene.gaussian_model3.transform import Resize, NormalizeImage, PrepareForNet


class NNSelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        B, N, C = x.shape
        x, _ = self.multihead_attn(x, x, x)
        return x


class NNCrossAttn(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, query):
        B, N, C = x.shape
        x, _ = self.multihead_attn(query, x, x)
        return x


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_head=8, act=nn.Sigmoid(), mlp_ratio=2.0):
        super().__init__()
        self.dim = in_dim
        self.num_head = num_head

        self.attn_norm = nn.LayerNorm(in_dim)
        self.attn = NNSelfAttn(in_dim, num_heads=num_head)

        self.mlp_norm = nn.LayerNorm(in_dim)
        self.mlp = FFN(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=out_dim)

        self.act = act
    

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.attn(self.attn_norm(x))
        x = self.mlp(self.mlp_norm(x))
        if self.act != None:
            x = self.act(x)
        x = x.squeeze(0)
        return x


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, num_head=8, act=nn.Sigmoid(), mlp_ratio=2.0):
        super().__init__()
        self.dim = in_dim
        self.num_head = num_head

        self.attn_norm = nn.LayerNorm(in_dim)
        self.attn_norm_query = nn.LayerNorm(in_dim)
        self.attn = NNCrossAttn(in_dim, num_heads=num_head)

        self.mlp_norm = nn.LayerNorm(in_dim)
        self.mlp = FFN(in_features=in_dim, hidden_features=int(in_dim*mlp_ratio), out_features=out_dim)

        self.act = act

    def forward(self, x, query):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.attn(self.attn_norm(x), self.attn_norm_query(query))
        x = self.mlp(self.mlp_norm(x))
        if self.act != None:
            x = self.act(x)
        x = x.squeeze(0)
        return x


class ImageFeatures(nn.Module):
    def __init__(self, in_dim=3, encoder_name='vitb'):
        super(ImageFeatures, self).__init__()
        self.encoder = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).cuda()
        self.features = {}
        def get_features(name):
            def hook(model, input, output):
                self.features[name] = output.detach()
            return hook
        self.encoder.layer1[2].conv2.register_forward_hook(get_features("1"))
        self.feat_dim = 64 # 256, 512, 1024
        self.downratio = 8

        self.seq_len = 1000

        self.in_dim = in_dim
        self.cross_attn = CrossAttentionBlock(self.feat_dim, self.in_dim, num_head=8, act=nn.Sigmoid(), mlp_ratio=2.0).to('cuda')
        self.preprocess_mlp = FFN(in_features=self.in_dim, hidden_features=self.feat_dim, out_features=self.feat_dim).to('cuda')

    def image_forward(self, image):
        x = self.encoder(image)
        return self.features["1"]
    
    def forward(self, image, xyz):
        image = F.interpolate(image.unsqueeze(0), scale_factor=1/self.downratio, mode='bilinear', align_corners=False)
        features = self.image_forward(image)
        # features = F.interpolate(features, scale_factor=self.downratio, mode='bilinear', align_corners=False)
        features = einops.rearrange(features, 'b c h w -> b (h w) c')
        xyz_feat = self.preprocess_mlp(xyz.unsqueeze(0))
        # xyz_offset = self.cross_attn(features, xyz_feat)
        # xyz.data += xyz_offset
        batch_num = xyz_feat.shape[0] // self.seq_len
        for i in range(batch_num):
            xyz_feat = xyz_feat[i*self.seq_len:(i+1)*self.seq_len]
            xyz_offset = self.cross_attn(features, xyz_feat)
            xyz[i*self.seq_len:(i+1)*self.seq_len].data += xyz_offset
        xyz_feat = xyz_feat[batch_num*self.seq_len:]
        xyz_offset = self.cross_attn(features, xyz_feat)
        xyz[batch_num*self.seq_len:].data += xyz_offset
        return xyz


# class ImageFeatures(nn.Module):
#     def __init__(self, in_dim=3, encoder_name='vitb'):
#         super(ImageFeatures, self).__init__()

#         # use dinov2 as encoder, supports {vits, vitb, vitl, vitg}
#         self.intermediate_layer_idx = {
#             'vits': [2, 5, 8, 11],
#             'vitb': [2, 5, 8, 11], 
#             'vitl': [4, 11, 17, 23], 
#             'vitg': [9, 19, 29, 39]
#         }
#         self.encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder_name), pretrained=True).to('cuda')
#         self.encoder_name = encoder_name
#         self.feat_dim = self.encoder.blocks[0].attn.qkv.in_features

#         self.in_dim = in_dim

#         self.cross_attn = CrossAttentionBlock(self.feat_dim, self.in_dim, num_head=8, act=nn.Sigmoid(), mlp_ratio=2.0).to('cuda')
#         self.preprocess_mlp = FFN(in_features=self.in_dim, hidden_features=self.feat_dim, out_features=self.feat_dim).to('cuda')


#     def image_forward(self, image_path, input_size=518):
#         x, (h, w) = self.image2tensor(image_path, input_size, color="BGR")
#         features = self.encoder.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder_name], return_class_token=False)
#         ### NOTE: something else
#         # patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14
#         # features = features.permute(0, 2, 1).reshape((features.shape[0], features.shape[-1], patch_h, patch_w))
#         return features
    
#     def forward(self, image_path, xyz):
#         features = self.image_forward(image_path)
#         xyz_feat = self.preprocess_mlp(xyz.unsqueeze(0))
#         features = features[-1]
#         xyz_offset = self.cross_attn(features, xyz_feat)
#         xyz.data += xyz_offset
#         return xyz

#     def image2tensor(self, image_path, input_size=518, color="RGB"):        
#         transform = Compose([
#             Resize(
#                 width=input_size,
#                 height=input_size,
#                 resize_target=False,
#                 keep_aspect_ratio=True,
#                 ensure_multiple_of=14,
#                 resize_method='lower_bound',
#                 image_interpolation_method=cv2.INTER_CUBIC,
#             ),
#             NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             PrepareForNet(),
#         ])
        
#         raw_image = cv2.imread(image_path)
#         h, w = raw_image.shape[:2]
        
#         if color == "RGB":
#             image = raw_image / 255.0
#         elif color == "BGR":
#             image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
#         else:
#             raise ValueError(f"color {color} not implemented")
        
#         image = transform({'image': image})['image']
#         image = torch.from_numpy(image).unsqueeze(0)
        
#         DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
#         image = image.to(DEVICE)
        
#         return image, (h, w)
