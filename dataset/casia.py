import os
import cv2
import random
import numpy as np
import torch.utils.data as data

class FACEDataset(data.Dataset):
    def __init__(self,
                 is_train :bool = False,
                 image_size : int = None,
                 data_dir :str = None, 
                 image_set :float = 0.9,
                 num_classes : int = 1,
                 seed: int = 1234) -> None:
        super().__init__()

        self.is_train = is_train
        self.image_set = image_set
        self.image_size = image_size
        self.num_classes = num_classes
        
        random.seed(seed)
        np.random.seed(seed)  # 设置随机种子

        self._imgs = []
        self._annos = []
        for index, class_name in enumerate(sorted(os.listdir(data_dir))):
            for img_name in sorted(os.listdir(os.path.join(data_dir, class_name))):
                self._imgs.append(os.path.join(
                    data_dir, class_name, img_name
                ))
                self._annos.append(index)
        
        if is_train:
            self._imgs = np.array(self._imgs[:int(image_set*len(self._imgs))])
            self._annos = np.array(self._annos[:int(image_set*len(self._annos))])
        else:
            self._imgs = np.array(self._imgs[int(image_set*len(self._imgs)):])
            self._annos = np.array(self._annos[int(image_set*len(self._annos)):])

    def __len__(self):
        return len(self._imgs)
    
    def __getitem__(self, index):
        img1, img2, img3, labels = self.load_image_target(index)

        images = self.transform(img1, img2, img3)

        return images, labels

    def load_image_target(self, index):
        # same guy
        class_idx = index//5 if self.is_train else index//5+self.num_classes*self.image_set
        imgs_path = self._imgs[self._annos == class_idx]

        if self.is_train:
            image_index = np.random.choice(range(0, len(imgs_path)), 2)
        else:
            image_index = np.array([2, 3])

        image1 = cv2.imread(imgs_path[image_index[0]])
        image2 = cv2.imread(imgs_path[image_index[1]])

        # different person
        diff_imgs_path = []
        while len(diff_imgs_path) < 1 or (diff_class_idx == class_idx):
            if self.is_train:
                diff_class_idx = np.random.choice(range(0, self.num_classes-1), 1)[0]
            else:
                if class_idx > self.num_classes*(self.image_set+0.05):
                    diff_class_idx = class_idx - 2
                else:
                    diff_class_idx = class_idx + 2

            diff_imgs_path = self._imgs[self._annos[:] == diff_class_idx]

        if self.is_train:
            image_index = np.random.choice(range(0, len(diff_imgs_path)), 1)
        else:
            image_index = np.array([1])

        diff_image = cv2.imread(diff_imgs_path[image_index[0]])

        labels = np.array([class_idx, class_idx, diff_class_idx])
        return image1, image2, diff_image, labels

    def transform(self, img1, img2, img3):
        images_list = [img1, img2, img3]
        images = np.zeros((3, 3, self.image_size, self.image_size))

        for index, image in enumerate(images_list):
            if random.random() < 0.5 and self.is_train:
                image = cv2.flip(image, 1)

            image = cv2.resize(image, [self.image_size, self.image_size])

            image = image.astype(np.float32)

            image /= 255.

            image = np.transpose(image, [2, 0, 1])

            images[index] = image

        return images