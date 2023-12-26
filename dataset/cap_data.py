import warnings

warnings.filterwarnings('ignore')
import os
from torch.utils.data import Dataset
from dataset.transforms import CVColorJitter, CVDeterioration, CVGeometry
from torchvision import transforms
import torch
import json


class ImageDataset(Dataset):
    def __init__(self,
                 json_file,
                 feats_dir,
                 is_rl,
                 is_training=True,
                 img_h=224,
                 img_w=224,
                 data_aug=True,
                 multiscales=True,
                 convert_mode='RGB',
                 ):
        json_file = json.load(open(json_file))
        self.data_aug = data_aug
        self.convert_mode = convert_mode
        self.is_rl = is_rl
        self.img_h = img_h
        self.img_w = img_w
        self.multiscales = multiscales
        self.is_training = is_training
        self.feats_dir = feats_dir
        self.folder = 'train2017' if is_training else 'val2017'
        self.captions = json_file['annotations']
        if self.data_aug:
            self.augment_tfs = transforms.Compose([
                CVGeometry(degrees=45, translate=(0.0, 0.0), scale=(0.5, 2.), shear=(45, 15), distortion=0.5, p=0.5),
                CVDeterioration(var=20, degrees=6, factor=4, p=0.25),
                CVColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1, p=0.25)
            ])
        self.totensor = transforms.ToTensor()

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        data = self.captions[idx]
        if not self.is_rl:
            image = os.path.join(self.feats_dir, f'{data["image_id"]:012d}.jpg.pth')
            image = torch.load(image, map_location='cpu')
        else:
            image = os.path.join(self.feats_dir, self.folder, f'{data["image_id"]:012d}.jpg')
        text = data['caption']
        return image, text


class ThaiCOCODataset(Dataset):
    def __init__(self,
        json_files,
        lst_dir,
        coco_dir,
        img_h=224,
        img_w=224,
    ):
        self.data = []
        self.img_h = img_h
        self.img_w = img_w
        for json_file in json_files:
            json_file = json.load(open(json_file))
            for key in json_file:
                for ann in json_file[key]:
                    if '-' in key.split('/')[0]:
                        self.data.append((os.path.join(lst_dir, key), ann))
                    else:
                        self.data.append((os.path.join(coco_dir, key+'.jpg'), ann))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, text = self.data[idx]
        return image, text


class MultiGenerationsDataset(Dataset):
    def __init__(self,
        json_files,
        lst_dir,
        coco_dir,
        img_h=224,
        img_w=224,
    ):
        self.data = []
        self.img_h = img_h
        self.img_w = img_w
        for json_file in json_files:
            json_file = json.load(open(json_file))
            for key in json_file:
                for ann in json_file[key]:
                    if '-' in key.split('/')[0]:
                        path = os.path.join(lst_dir, key)
                    else:
                        path = os.path.join(coco_dir, key+'.jpg')
                    le = len(ann)
                    if le  == 1:
                        self.data.append(
                            (path, ann[0], 1)
                        )
                        continue
                    for i, text in enumerate(ann):
                        self.data.append(
                            (path, text, i / (le-1))
                        )
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, text, val = self.data[idx]
        return image, text, val


if __name__ == '__main__':
    src_dir = "/home/palm/data/coco/images"
    val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
    train_set = ImageDataset(
        val_json,
        src_dir=src_dir,
    )
    train_set[0]
