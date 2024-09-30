import torch
import numpy as np
import torch.nn.functional as F
   
def triplet_loss(alpha = 0.2):
    def _triplet_loss(pred, batch):
        anchor = pred[:batch]
        negative = pred[2*batch:]
        positive = pred[batch:2*batch]

        pos_dist = torch.sqrt(torch.sum(torch.pow(anchor - positive,2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(anchor - negative,2), axis=-1))

        keep_all = (neg_dist - pos_dist < alpha).cpu().numpy().flatten()
        hard_triplets = np.where(keep_all == 1)

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))

        return loss

    return _triplet_loss

def build_loss(args, device, loss_cfg):
    loss_dic = {
        'triplet_loss': triplet_loss()
    }
    
    return loss_dic[loss_cfg]