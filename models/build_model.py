import torch

def build_model_pu(args, ema=False, try_assert=True, old_etf=None):
    if args.dataset in ['wisig', 'oracle', 'lora']:
        from . import resnet_cifar as models    
    
    model = models.resnet18_pu(
        no_class=args.no_class, 
        try_assert=try_assert, 
        old_etf=old_etf
    )
    
    if ema:
        for param in model.parameters():
            param.detach_()

    return model