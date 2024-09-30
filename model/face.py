import os
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    from model.backbone.mobilenet import MobileNetV1
    from model.backbone.inception import InceptionV1
except:
    import sys
    sys.path.append('../')
    from config import parse_args
    from backbone.mobilenet import MobileNetV1
    from backbone.inception import InceptionV1
    
# ImageNet-1K pretrained weight
model_urls = {
    "mobilenet_v1": "https://github.com/EricReno/Facenet/releases/download/weights/mobilenet_v1.pth",
    "inception_v1": "https://github.com/EricReno/Facenet/releases/download/weights/inception_v1.pth",
}

class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", 
                 dropout_keep_prob=0.5, 
                 embedding_size=128, 
                 num_classes=None, 
                 pretrained=False):
        super(Facenet, self).__init__()
        
        self.trainable = True

        if backbone == "mobilenet_v1":
            self.backbone = MobileNetV1(pretrained)
            flat_shape = 1024
        elif backbone == "inception_v1":
            self.backbone = InceptionV1(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        
        self.avg        = nn.AdaptiveAvgPool2d((1,1))
        self.Dropout    = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn    = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if self.trainable:
            self.classifier = nn.Linear(embedding_size, num_classes)

        if pretrained:
            state_dict = torch.hub.load_state_dict_from_url(
                model_urls[backbone], 
                model_dir="deploy", progress=True)
            self.backbone.load_state_dict(state_dict)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        
        x = F.normalize(before_normalize, p=2, dim=1)

        if not self.trainable:
            return x
        else:
            cls = self.classifier(before_normalize)
            return x, cls

if __name__ == "__main__":
    from thop import profile
    args = parse_args()

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input = torch.randn(3, 3, args.image_size, args.image_size).to(device)

    model = Facenet(backbone=args.backbone, 
                    num_classes=len(os.listdir(args.data_root)), 
                    pretrained=True).to(device)

    outputs = model(input)
    print(model)
    
    flops, params = profile(model, inputs=(input, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))