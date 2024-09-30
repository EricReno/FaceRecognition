import os
import torch
import numpy as np

from config import parse_args
from utils.flops import compute_flops
from model.build import build_facenet
from dataset.build import build_dataset

class Evaluator():
    """ LFW Evaluation class"""
    def __init__(self,
                 device,
                 dataset,
                 visualization) -> None:
        
        self.device = device
        self.dataset = dataset
        self.visualization = visualization
 
    def inference(self, model):
        labels, distances = [], []

        for i in range(len(self.dataset)):
            images, label = self.dataset[i]
            with torch.no_grad():
                image_a =  torch.from_numpy(images[0]).float().unsqueeze(0).to(self.device)
                image_p =  torch.from_numpy(images[1]).float().unsqueeze(0).to(self.device)
                image_n =  torch.from_numpy(images[2]).float().unsqueeze(0).to(self.device)

                out_a, out_p, out_n = model(image_a), model(image_p), model(image_n)

                dist_ap = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
                dist_an = torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))
                dist_pn = torch.sqrt(torch.sum((out_p - out_n) ** 2, 1))

            labels.extend([True, False, False])
            distances.extend([dist_ap.data.cpu().numpy()[0],
                              dist_an.data.cpu().numpy()[0],
                              dist_pn.data.cpu().numpy()[0],
                              ])
            
            print('Inference: {} / {}'.format(i+1, len(self.dataset)), end='\r')

        labels    = np.array(labels)
        distances = np.array(distances)

        return labels, distances        
        
    def calculate_accuracy(self, threshold, dist, actual_issame):
        predict_issame = np.less(dist, threshold)
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

        tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
        fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
        acc = float(tp+tn)/dist.size
        return tpr, fpr, acc

    def eval(self, model, epoch):        
        labels, distances = self.inference(model)
        assert(len(labels) == len(distances))
        thresholds = np.arange(0, 3, 0.01)

        accuracy = np.zeros(len(thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            predict_issame = np.less(distances, threshold)
            tp = np.sum(np.logical_and(predict_issame, labels))
            fp = np.sum(np.logical_and(predict_issame, np.logical_not(labels)))
            tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(labels)))
            fn = np.sum(np.logical_and(np.logical_not(predict_issame), labels))

            tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
            fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
            accuracy[threshold_idx] = float(tp+tn)/distances.size

        print('')
        print('~~~~~~~~')
        print('Max Acc = %2.5f, threshold = %2.5f' %(np.max(accuracy), thresholds[np.argmax(accuracy)]))
        print('~~~~~~~~')
        print('')

        return np.max(accuracy)

def build_eval(args, dataset, device):
    evaluator = Evaluator(
        device   =device,
        dataset  = dataset,
        visualization = args.eval_visualization)
    
    return evaluator
    
if __name__ == "__main__":
    args = parse_args()
    args.resume_weight_path = "None"
    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('use cuda')
    else:
        device = torch.device('cpu')

    num_classes = len(os.listdir(args.data_root))
    val_dataset = build_dataset(args, is_train=False, num_classes=num_classes)

    model = build_facenet(args, device, num_classes)
    compute_flops(model, args.image_size, device)

    model.eval()
    model.trainable = False
    state_dict = torch.load(f = os.path.join('deploy', args.model_weight_path), 
                            map_location = 'cpu',
                            weights_only = False)
    model.load_state_dict(state_dict["model"])
    print('Epoch:', state_dict['epoch'])
    print('Acc:', state_dict['Acc'])

    # VOC evaluation
    evaluator = build_eval(args, val_dataset, device)
    acc = evaluator.eval(model, state_dict['epoch'])