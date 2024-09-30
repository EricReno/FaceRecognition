import torch
import numpy as np
from .casia import FACEDataset

# DataLoader中collate_fn使用
def collate_fn(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)
    
    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    labels  = torch.from_numpy(np.array(labels)).long()
    return images, labels

def build_dataset(args, is_train, num_classes):
    dataset = FACEDataset(is_train = is_train,
                           image_size=args.image_size, 
                           data_dir = args.data_root,
                           image_set = args.image_set,
                           num_classes = num_classes
                           )
    return dataset

def build_dataloader(args, dataset):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size//3,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return dataloader