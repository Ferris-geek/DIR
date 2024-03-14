import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

class FreiburgBase(Dataset):
    def __init__(self,
                 txt_file_ir,
                 txt_file_rgb,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 val=False
                 ):
        self.ir_data_paths = txt_file_ir
        self.rgb_data_paths = txt_file_rgb
        self.data_root = data_root
        self.val=val
        
        with open(self.ir_data_paths, "r") as f:
            self.ir_image_paths = f.read().splitlines()
        with open(self.rgb_data_paths, "r") as f:
            self.rgb_image_paths = f.read().splitlines()
        
        self.ir_image_paths  = sorted(self.ir_image_paths)
        self.rgb_image_paths = sorted(self.rgb_image_paths)
        
        assert len(self.ir_image_paths) == len(self.rgb_image_paths)
        self._length = len(self.ir_image_paths)
        self.labels = {
            "ir_file_path_": [os.path.join(self.data_root, l)
                           for l in self.ir_image_paths],
            "vi_file_path_": [os.path.join(self.data_root, l)
                           for l in self.rgb_image_paths],
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
        center = random.randint(635, 1285)
        example = dict((k, self.labels[k][i]) for k in self.labels)
        
        # preprocessing ir imgs
        image_ir = cv2.imread(example["ir_file_path_"])
        if image_ir is None:
            print(example["ir_file_path_"])
            
        img_ir = np.array(image_ir).astype(np.uint8)
        crop = min(img_ir.shape[0], img_ir.shape[1])
        h, w, = img_ir.shape[0], img_ir.shape[1]
        img_ir = img_ir[0:crop, (center - crop//2):(center + crop// 2)]

        image_ir = Image.fromarray(img_ir)
        if self.size is not None:
            image_ir = image_ir.resize((self.size, self.size), resample=self.interpolation)
        
        image_ir = np.array(image_ir).astype(np.uint8)
        example["image"] = (image_ir / 127.5 - 1.0).astype(np.float32)
        
        # preprocesing visible imgs
        image_vi = Image.open(example["vi_file_path_"])
        if not image_vi.mode == "RGB":
            image_vi = image_vi.convert("RGB")

        img_vi = np.array(image_vi).astype(np.uint8)
        crop = min(img_vi.shape[0], img_vi.shape[1])
        img_vi = img_vi[0:crop, (center - crop//2):(center + crop// 2)]

        image_vi = Image.fromarray(img_vi)
        if self.size is not None:
            image_vi = image_vi.resize((self.size, self.size), resample=self.interpolation)
        
        image_vi = np.array(image_vi).astype(np.uint8)
        
        image_cond = image_vi
            
        example["conditional"] = (image_cond / 127.5 - 1.0).astype(np.float32)
        
        return example


class FreiburgTrain(FreiburgBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file_ir="./data/Freiburg/Freiburg_ir_train.txt", 
                         txt_file_rgb="./data/Freiburg/Freiburg_rgb_train.txt", 
                         data_root="/path/to/data", 
                         **kwargs)


class FreiburgVal(FreiburgBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file_ir="./data/Freiburg/Freiburg_ir_test.txt", 
                         txt_file_rgb="./data/Freiburg/Freiburg_rgb_test.txt", 
                         data_root="/path/to/data",
                         val=True,
                         flip_p=flip_p, **kwargs)
