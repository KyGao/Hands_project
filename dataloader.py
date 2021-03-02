import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import random

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def my_transform(rgb, depth):
    left = random.randint(0, 74)
    rgb = transforms.functional.crop(rgb, 0, left, 224, 224)
    depth = transforms.functional.crop(depth, 0, left, 224, 224)
    if random.random() > 0.5:
        rgb = transforms.functional.hflip(rgb)
        depth = transforms.functional.hflip(depth)
    if random.random() > 0.5:
        rgb = transforms.functional.vflip(rgb)
        depth = transforms.functional.vflip(depth)
    return rgb, depth

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)
        w, h = img.size
        w2 = int(w/2)
        rgb = img.crop((0, 0, w2, h))
        depth = img.crop((w2, 0, w, h))


        if self.data_transforms is not None:
            try:
                if self.dataset == 'val':
                    rgb = self.data_transforms[self.dataset](rgb)
                    depth = self.data_transforms[self.dataset](depth)
                elif self.dataset == 'train':
                    rgb, depth = my_transform(rgb, depth)
                    rgb = self.data_transforms[self.dataset](rgb)
                    depth = self.data_transforms[self.dataset](depth)
            except:
                print("Cannot transform image: {}".format(img_name))
        return rgb, depth, label

def get_data_loader(data_path, batch_size):
    print("Init data transforms......")
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomSizedCrop(224),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            #transforms.Scale(256),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Load dataset......")
    image_datasets={}
    image_datasets['train'] = customData(img_path=data_path + "/dat/train",
                                         txt_path=data_path + "/label/train.txt",
                                         data_transforms=data_transforms,
                                         dataset='train')
    image_datasets['val'] = customData(img_path=data_path + "/dat/val",
                                       txt_path=data_path + "/label/val.txt",
                                       data_transforms=data_transforms,
                                       dataset='val')

    # wrap your data and label into Tensor
    print("wrap data into Tensor......")
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=batch_size,
                                                 shuffle=True) for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("total dataset size:", dataset_sizes)

    return dataloders