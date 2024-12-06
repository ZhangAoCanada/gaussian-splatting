import torch
from scene.PTv3.feature_predictor import FeaturePredictor

LR_DICT = {
    "base": 1e-5, #Useless
    "embedding": 1e-5,
    "backbone": 1e-5,
    "features_dc": 1e-5,
    "features_rest": 1e-6,
    "scales": 1e-5,
    "opacities": 1e-5,
    "quats": 1e-5,
    "means": 1e-5
}

def build_3DGSoptimizer(gs_params, 
                        lr_dict=LR_DICT, 
                        optimizer_type='adam', 
                        optimizer_params={'eps': 1e-15}):
    params_lr = []
    for param in gs_params:
        lr = lr_dict.get(param, lr_dict['base'])
        params_lr.append({'params': gs_params[param], 'lr': lr})
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(params_lr, 
                                     lr = lr_dict['base'],
                                     **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(params_lr, 
                                    lr = lr_dict['base'])
    else:
        raise NotImplementedError
    return optimizer


def build_optimizer(model, 
                    lr_dict=LR_DICT,
                    optimizer_type='adam',
                    optimizer_params={'eps': 1e-15}):  
    params_lr = []
    if type(model) == FeaturePredictor:
        if model.backbone_type != 'empty':
            params_lr.append({'params': model.backbone.parameters(), 'lr': lr_dict['backbone']})
        for feature in model.features_outputhead.keys():
            lr = lr_dict.get(feature, lr_dict['base'])
            params_lr.append({'params': model.features_outputhead[feature].parameters(), 'lr': lr})
    else:
        for param in model.parameters():
            lr = lr_dict.get(param, lr_dict['base'])
            params_lr.append({'params': param, 'lr': lr})

    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(params_lr, 
                                     lr = lr_dict['base'],
                                     **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(params_lr, 
                                    lr = lr_dict['base'])
    else:
        raise NotImplementedError
    return optimizer


def build_scheduler(optimizer, 
                    total_step, 
                    schedule='constant', 
                    warmup_step=0):
    if schedule == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1)
    elif schedule == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1-step/total_step)
    elif schedule == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_step)
    elif schedule == 'exponential':
        raise ValueError
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exponential_gamma)
    else:
        raise NotImplementedError
    if warmup_step > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step/warmup_step)
        lr_scheduler = torch.optim.lr_scheduler.ChainedScheduler([warmup_scheduler, lr_scheduler], optimizer)
    return lr_scheduler