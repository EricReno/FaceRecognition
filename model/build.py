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

    # # keep training
    # if args.resume_weight_path and args.resume_weight_path != "None":
    #     ckpt_path = os.path.join('deploy', args.resume_weight_path)
    #     checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    #     # checkpoint state dict
    #     try:
    #         checkpoint_state_dict = checkpoint['model']
    #         print('Load model from the checkpoint: ', ckpt_path)
    #         model.load_state_dict(checkpoint_state_dict, strict=False)
            
    #         del checkpoint, checkpoint_state_dict
    #     except:
    #         print("No model in the given checkpoint.")

    return model

        # model_dict      = model.state_dict()
        # pretrained_dict = torch.load(os.path.join('deploy', args.resume_weight_path), map_location = device)
        # load_key, no_load_key, temp_dict = [], [], {}
        # for k, v in pretrained_dict.items():
        #     if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        #         temp_dict[k] = v
        #         load_key.append(k)
        #     else:
        #         no_load_key.append(k)
        # model_dict.update(temp_dict)
        # model.load_state_dict(model_dict)
        
        # print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        # print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        # print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")