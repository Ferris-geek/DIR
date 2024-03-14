import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torchvision.transforms.functional as tf

class KAISTBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        
        self.ir_data_root = os.path.join(self.data_root, 'lwir')
        self.vi_data_root = os.path.join(self.data_root, 'visible')
        
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "ir_file_path_": [os.path.join(self.ir_data_root, l)
                           for l in self.image_paths],
            "vi_file_path_": [os.path.join(self.vi_data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        # preprocessing ir imgs
        image_ir = Image.open(example["ir_file_path_"])
        if not image_ir.mode == "RGB":
            image_ir = image_ir.convert("RGB")

        img_ir = np.array(image_ir).astype(np.uint8)
        crop = min(img_ir.shape[0], img_ir.shape[1])
        h, w, = img_ir.shape[0], img_ir.shape[1]
        img_ir = img_ir[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image_ir = Image.fromarray(img_ir)
        if self.size is not None:
            image_ir = image_ir.resize((self.size, self.size), resample=self.interpolation)

        # preprocesing visible imgs
        image_vi = Image.open(example["vi_file_path_"])
        if not image_vi.mode == "RGB":
            image_vi = image_vi.convert("RGB")

        img_vi = np.array(image_vi).astype(np.uint8)
        crop = min(img_vi.shape[0], img_vi.shape[1])
        h, w, = img_vi.shape[0], img_vi.shape[1]
        img_vi = img_vi[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image_vi = Image.fromarray(img_vi)
        if self.size is not None:
            image_vi = image_vi.resize((self.size, self.size), resample=self.interpolation)
        
        image_cond = image_vi
        
        image_ir = np.array(image_ir).astype(np.uint8)
        image_vi = np.array(image_vi).astype(np.uint8)
        example["image"] = (image_ir / 127.5 - 1.0).astype(np.float64)
        example["conditional"] = (image_vi / 127.5 - 1.0).astype(np.float64)
        
        return example


class KAISTTrain(KAISTBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/kaist/KAIST_00_01_03_04.txt", 
                         data_root="/path/to/data", 
                         **kwargs)


class KAISTVal(KAISTBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/kaist/KAIST_02_05.txt", 
                         data_root="/path/to/data",
                         flip_p=flip_p, **kwargs)
