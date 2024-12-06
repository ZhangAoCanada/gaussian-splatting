import utils
from .transform_utils import remove_outliers, MinMaxScaler
import torch
import torch.nn as nn
from collections import OrderedDict
from .pointtransformer_v3 import PointTransformerV3Model
from .spconv import SparseConvModel
from typing import List

FEATURE2CHANNEL = {
    'means': 3,
    'features_dc': 3,
    'features_rest': 3,
    'opacities': 1,
    'scales': 3,
    'quats': 4,
}
ALL_FEATURES = ['means','features_dc','features_rest','opacities','scales','quats']

class FeaturePredictor(nn.Module):
    def __init__(self, 
                 backbone_type="PT",
                 sh_degree=1,
                 input_features=['means','scales', 'opacities', 'quats', 'features_dc', 'features_rest'],
                 input_feat_to_mlp=True,
                 output_features=['means','scales', 'opacities', 'quats', 'features_dc', 'features_rest'],
                 output_head_nlayer=4,
                 output_head_type='mlp-relu',
                 output_head_width=128,
                 output_features_type='res', # 'dc:direct component or res:residual"
                 res_feature_activation={
                    'means': nn.Tanh(),
                    'features_dc': nn.Identity(),
                    'features_rest': nn.Identity(),
                    'scales': nn.Identity(),
                    'opacities': nn.Identity(),
                    'quats': nn.Identity(),
                 },
                 max_scale_normalized=1e-2,
                 grid_resolution=384,
                 resume_ckpt=None,
                 input_embed_to_mlp=False,
                 zeroinit=True,
                 ):
        super(FeaturePredictor, self).__init__()
        self.sh_degree = sh_degree
        sh_dim = (sh_degree+1)**2-1
        FEATURE2CHANNEL['features_rest'] = sh_dim*3
        self.input_features = input_features
        self.input_feat_to_mlp = input_feat_to_mlp
        in_channels = sum([FEATURE2CHANNEL[feature] for feature in input_features])
        self.gs_features_dim = in_channels
        self.output_features = output_features
        if max_scale_normalized<=0:
            print('Setting max_scale_normalized <0, turning off scale clamping')
        self.max_scale_normalized = max_scale_normalized
        self.backbone_type = backbone_type
        self.grid_resolution = grid_resolution
        self.resume_ckpt = resume_ckpt
        self.output_features_type = output_features_type 
        self.res_feature_activation = res_feature_activation 
        self.input_embed_to_mlp = input_embed_to_mlp

        if backbone_type == 'SP':
            self.backbone = SparseConvModel(in_channels=in_channels)
        elif backbone_type == 'PT':
            self.backbone = PointTransformerV3Model(in_channels=in_channels)
        else:
            raise NotImplementedError
        head_input_dim = self.backbone.output_dim
        if self.input_feat_to_mlp:
            head_input_dim += in_channels

        self.features_outputhead = nn.ModuleDict()
        for feature in output_features:
            if output_head_type=='mlp-relu':
                module_list = nn.ModuleList()
                for _ in range(output_head_nlayer-1):
                    module_list.extend(
                        [nn.Linear(head_input_dim if _==0 else output_head_width, output_head_width),
                        nn.ReLU()]
                    )
                outputdim_ = FEATURE2CHANNEL[feature]
                module_list.append(
                    nn.Linear(output_head_width if output_head_nlayer>1 else head_input_dim, outputdim_)
                )
                self.features_outputhead[feature] = nn.Sequential(*module_list)
            else:
                raise NotImplementedError
        if zeroinit:
            #init the last layer of each feature predictor to be zeros
            for k, module in self.features_outputhead.items():
                module[-1].weight.data.zero_()
                module[-1].bias.data.zero_()
    
    def normalized_gs(self, batch_gs):
        scalers = []
        batch_normalized_gs = []
        for gs in batch_gs:
            normalized_gs = {}
            scaler = MinMaxScaler()
            scaler.fit(gs['means'])
            normalized_gs['means'] = scaler.transform(gs['means']) 
            normalized_gs['scales'] = gs['scales'] + torch.log(scaler.scale_)
            normalized_gs['features_dc'] = gs['features_dc']
            scalers.append(scaler)
            batch_normalized_gs.append(normalized_gs)
        return batch_normalized_gs, scalers

    def unnormalized_gs(self, batch_gs, scalers): #TODO
        batch_unnormalized_gs = []
        for gs, scaler in zip(batch_gs, scalers):
            unnormalized_gs = {}
            for key in gs:
                if key=='means': #The predicted gs may not contain means
                    unnormalized_gs['means'] = scaler.inverse_transform(gs['means'])
                elif key=='scales':
                    unnormalized_gs['scales'] = gs['scales'] - torch.log(scaler.scale_)
                else:
                    unnormalized_gs[key] = gs[key]
            batch_unnormalized_gs.append(unnormalized_gs)
        return  batch_unnormalized_gs

    # def forward(self, batch_gs):
    #     #1. Normalize
    #     batch_normalized_gs, batch_scalers = self.normalized_gs(batch_gs) #Move to dataloader part
    def forward(self, batch_normalized_gs: List, batch_scene_idx: List, 
                **kwargs):
        # start = time()
        device = batch_normalized_gs[0]['means'].device #It should be cuda
        input_keys = sorted(batch_normalized_gs[0])

        #2. Batchify
        offset = torch.tensor([gs['means'].shape[0] for gs in batch_normalized_gs]).cumsum(0)
        feat = []
        
        for bi, (gs, idx) in enumerate(zip(batch_normalized_gs, batch_scene_idx)):
            feat_list = []
            for key in self.input_features:
                if key=='means':
                    feat_list.append(gs[key])
                elif key == 'features_rest' or key == "features_dc":
                    feat_list.append(gs[key].view(gs[key].shape[0], -1))
                else:
                    feat_list.append(gs[key])
            feat.append(torch.cat(feat_list, dim=1)) #N, D
        feat = torch.cat(feat, dim=0) #Bx-N, D

        if self.backbone_type in ['PT','SP']:
            model_input = {
                'coord': torch.cat([gs['means'] for gs in batch_normalized_gs], dim=0),
                'grid_size': torch.ones([3])*1.0/self.grid_resolution,
                'offset': offset.to(device),
                'feat': feat,
            }
            model_input['grid_coord'] = torch.floor(model_input['coord']*self.grid_resolution).int() #[0~1]/
        else:
            raise NotImplementedError

        y = self.backbone(model_input)

        if self.backbone_type in ['PT']:
            y = y['feat']

        hidden_features = y
        if self.input_feat_to_mlp:
            y = torch.cat([y, feat], dim=1)
    
        output = OrderedDict()
        for feature in self.output_features:
            feature_o = self.features_outputhead[feature](y)
            if self.output_features_type=='dc': #Predict the feature itself
                if feature == 'scales' and self.max_scale_normalized>0:
                    feature_o = torch.nn.functional.relu(feature_o)*-1
                    feature_o = feature_o + torch.log(torch.tensor(self.max_scale_normalized))
                if feature=='features_rest' or key == "features_dc":
                    feature_o = feature_o.view(feature_o.shape[0], -1, 3)
                output[feature] = feature_o
            elif self.output_features_type=='res': #Predict the modulation and residual (mod first and res then)
                pointer = 0
                feature_o_res = feature_o[:, pointer:pointer+FEATURE2CHANNEL[feature]]
                ###########################################################
                if feature == "scales":
                    feature_o_res = torch.nn.functional.tanh(feature_o_res)
                    # feature_o_res = torch.nn.functional.relu(feature_o_res) * -1
                    # feature_o_res = torch.nn.functional.mish(feature_o_res) * -1
                    # feature_o_res = torch.nn.functional.gelu(feature_o_res) * -1
                ###########################################################
                feature_o_res = self.res_feature_activation[feature](feature_o_res)
                pointer += FEATURE2CHANNEL[feature]
                if feature == 'features_rest' or key == "features_dc":
                    feature_o_res = feature_o_res.view(feature_o_res.shape[0], -1, 3)
                ###########################################################
                if feature == "means":
                    feature_o_res = feature_o_res * 0.1
                ###########################################################
                output[feature] = feature_o_res

        #-2. Unbatchify
        out_batch_normalized_gs = []
        if self.backbone_type in ['PT','SP']:
            left = 0
            for ii,(right, in_gs) in enumerate(zip(offset, batch_normalized_gs)):
                out_normalized_gs = {}
                for feature in self.output_features:
                    if self.output_features_type=='dc':
                        out_normalized_gs[feature] = output[feature][left:right]
                    elif self.output_features_type=='res':
                        out_normalized_gs[feature] = in_gs[feature] + output[feature][left:right] #Residual
                out_batch_normalized_gs.append(out_normalized_gs)
                left = right

        for key in ALL_FEATURES:
            if self.sh_degree==0 and key=='features_rest':
                continue
            if key not in self.output_features: #If the feature is not in the output, we need to copy it
                for out_gs, in_gs in zip(out_batch_normalized_gs, batch_normalized_gs):
                    out_gs[key] = in_gs[key]

        assert len(out_batch_normalized_gs) == 1, 'Now only support batch size 1'
        return out_batch_normalized_gs

