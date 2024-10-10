import os
import torch
import torch.nn as nn
from .face import Facenet

def build_facenet(args, device, num_classes):
    print('==============================')
    print('Build Model ...')
    print('Backbone: {}'.format(args.backbone))

    model = Facenet(
        backbone=args.backbone, 
        num_classes=num_classes, 
        pretrained=False).to(device)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03

    # keep training
    if args.resume_weight_path and args.resume_weight_path != "None":
        ckpt_path = os.path.join('deploy', args.resume_weight_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # checkpoint state dict
        try:
            checkpoint_state_dict = checkpoint['model']
            print('Load model from the checkpoint: ', ckpt_path)
            model.load_state_dict(checkpoint_state_dict, strict=False)
            
            del checkpoint, checkpoint_state_dict
        except:
            print("No model in the given checkpoint.")

    return model