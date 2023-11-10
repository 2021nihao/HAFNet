import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split

import torch
import torch.utils.data as data
from torchvision import transforms
from toolbox.datasets.augmentations import Resize, Compose, ColorJitter, RandomHorizontalFlip, RandomCrop, RandomScale, \
    RandomRotation


class MirrorRGBD(data.Dataset):

    def __init__(self, cfg, mode='trainval', do_aug=True):

        assert mode in ['train', 'val', 'trainval', 'test', 'test_day', 'test_night'], f'{mode} not support.'
        self.mode = mode

        ## pre-processing 数据集图片的预处理
        # 定于转换方式，将几个操作组合起来
        self.im_to_tensor = transforms.Compose([
            # 归一化到(0,1)  (C,H,W)
            transforms.ToTensor(),
            # nb到(-1,1)   (x-mean)/std
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.dp_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
            # transforms.Normalize([0.449], [0.226]),
        ])

        # 数据集所在目录
        self.root = cfg['root']
        self.n_classes = cfg['n_classes']

        scale_range = tuple(float(i) for i in cfg['scales_range'].split(' '))
        # 裁切尺寸
        crop_size = tuple(int(i) for i in cfg['crop_size'].split(' '))

        self.aug = Compose([
            ColorJitter(
                # 亮度
                brightness=cfg['brightness'],
                # 对比度
                contrast=cfg['contrast'],
                # 饱和度
                saturation=cfg['saturation']),
            # 依据概率p对图片进行水平翻转
            RandomHorizontalFlip(cfg['p']),
            RandomScale(scale_range),
            # 依据给定size随机剪裁
            RandomCrop(crop_size, pad_if_needed=True)
        ])

        # self.aug1 = Compose([
        #     RandomScale(scale_range),   #获取剪裁的比例
        #     # 依据给定size随机剪裁
        #     RandomCrop(crop_size, pad_if_needed=True)
        # ])

        # 解开封禁
        self.val_resize = Resize(crop_size)

        self.mode = mode
        self.do_aug = do_aug

        if cfg['class_weight'] == 'enet':
            self.class_weight = np.array(
                [1.5105, 16.6591, 29.4238, 34.6315, 40.0845, 41.4357, 47.9794, 45.3725, 44.9000])
            self.binary_class_weight = np.array([1.5121, 10.2388])
        # 解决图像分割中样本不平衡问题
        elif cfg['class_weight'] == 'median_freq_balancing':
            self.class_weight = np.array(
                [0.0118, 0.2378, 0.7091, 1.0000, 1.9267, 1.5433, 0.9057, 3.2556, 1.0686])
            self.binary_class_weight = np.array([0.5454, 6.0061])
        else:
            raise (f"{cfg['class_weight']} not support.")

        # 读取训练集测试集目录
        with open(os.path.join(self.root, f'{mode}.txt'), 'r') as f:
            self.infos = f.readlines()

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        image_path = self.infos[index].strip()
        image = Image.open(os.path.join(self.root, self.mode, 'image', image_path + '.jpg'))
        # depth = Image.open(os.path.join(self.root, self.mode, 'depth', image_path + '.png'))
        depth = Image.open(os.path.join(self.root, self.mode, 'depth', image_path + '.png'))
        depth = Image.fromarray(np.uint8(depth))
        # print(image_path)
        # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
        # print(image_path + "ok")
        # depth = Image.fromarray(depth).convert('RGB')
        # depth = Image.fromarray(depth)
        # print('111', depth[:5,:5])
        # depth_uint16 = io.imread(os.path.join(self.root, self.mode, 'depth', image_path + '.png'))
        # depth = ( depth_uint16 / 65535).astype(np.float32)   #归一化  0-1之间
        # depth = Image.fromarray(depth)
        # print("333", depth)

        # depth_normalized = Image.open(os.path.join(self.root, self.mode, 'depth_normalized', image_path + '.png')).convert('RGB')
        # depth = np.load(os.path.join(self.root, self.mode, 'depth', image_path + '.npy'))
        # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
        # depth = Image.fromarray(depth).convert('RGB')
        label = Image.open(os.path.join(self.root, self.mode, 'mask_single', image_path + '.png')).convert('L')
        bound = Image.open(os.path.join(self.root, self.mode, 'Boundary', image_path + '.png'))
        # binary_label = Image.open(os.path.join(self.root, self.mode, 'GT', image_path + '.png'))

        sample = {
            'image': image,
            'depth': depth,
            'label': label,
            # 'depth_normalized':depth_normalized,

            'bound': bound,
            # 'binary_label': binary_label,
        }

        sample = self.val_resize(sample)

        if self.mode in ['train', 'trainval'] and self.do_aug:
            sample = self.aug(sample)
        # 因为某些模型需要，将测试集也进行增强
        # elif self.mode in ['test']:
        #     sample = self.val_resize(sample)
        sample['image'] = self.im_to_tensor(sample['image'])    #标准化处理过后,[-1.9+, 2.39+]
        sample['depth'] = self.dp_to_tensor(sample['depth'])   #标准化处理过后,[-1.9+, 2.39+]
        # print("333", sample['depth'], sample['depth'].shape)
        sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64) / 255. ).long()
        # sample['depth_normalized'] = torch.from_numpy(np.asarray(sample['depth_normalized'], dtype=np.int64)).long()

        # if int(image_path) == 2551:
        # if int(image_path) == 2504 :
        #         sample['num'] = image_path
        #
        #         print('111_image',image_path, sample['label'], sample['label'].shape)
        # print('111_depth', sample['depth'], sample['depth'].shape)
        # print('111_label', sample['label'], sample['label'].shape)

        # sample['label'] = torch.from_numpy(np.asarray(sample['label'], dtype=np.int64) /255.).long()
        sample['bound'] = torch.from_numpy(np.asarray(sample['bound'], dtype=np.int64) / 255.).long()
        # sample['binary_label'] = torch.from_numpy(np.asarray(sample['binary_label'], dtype=np.int64) / 255.).long()
        sample['label_path'] = image_path.strip().split('/')[-1] + '.png'  # 后期保存预测图时的文件名和label文件名一致
        return sample

    # @property
    # def cmap(self):
    #     return [
    #         (0, 0, 0),  # unlabelled
    #         (64, 0, 128),  # car
    #         (64, 64, 0),  # person
    #         (0, 128, 192),  # bike
    #         (0, 0, 192),  # curve
    #         (128, 128, 0),  # car_stop
    #         (64, 64, 128),  # guardrail
    #         (192, 128, 128),  # color_cone
    #         (192, 64, 0),  # bump
    #     ]
    @property
    def cmap(self):
        return [
            (0, 0, 0),  # unlabelled
            (255, 255, 255)
        ]


if __name__ == '__main__':
    root = '/home/noone/桌面/models/RGBD-Mirror'
    mode = 'train'
    with open(os.path.join(root, f'{mode}.txt'), 'r') as f:
        infos = f.readlines()
    for index in range(len(infos)):
        image_path = infos[index].strip()
        # image = Image.open(os.path.join(root, mode, 'image', image_path + '.jpg'))
        # image = Image.open(os.path.join(root, mode, 'depth', image_path + '.png'))
        # image = (np.array(image) - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = io.imread(os.path.join(root, mode, 'depth', image_path + '.png'))
        image = (image / 65535).astype(np.float32)
        image = Image.fromarray(image)

        # depth_uint16 = io.imread(os.path.join(root, mode, 'depth', image_path + '.png'))
        # depth = (depth_uint16 / 65535).astype(np.float32)
        # depth = Image.fromarray(depth)

        # depth = np.load(os.path.join(root    , mode, 'T', image_path + '.npy'))
        # depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth)) * 255
        # depth = Image.fromarray(depth).convert('RGB')
        # label = Image.open(os.path.join(root, mode, 'GT', image_path + '.png'))
        # bound = Image.open(os.path.join(root, mode, 'Boundary', image_path + '.png'))
        # if index == 1:
        im_to_tensor = transforms.Compose([
                # 归一化到(0,1)
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                # nb到(-1,1)
            # transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        dep_to_norm = transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),


                # print(label.shape)

        image = im_to_tensor(image)
        print("111", image, image.shape)
            # depth = dep_to_norm(depth)
            # print("222", image)
            # print(image_path)
            # print(image.shape)
            # import matplotlib.pyplot as plt
            # plt.imshow(depth)
            # plt.show()
        # print(image.shape)
        # print(depth.size)
    # import json
    #
    # path = '/home/dtrimina/Desktop/lxy/Segmentation_final/configs/cccmodel/irseg_cccmodel.json'
    # with open(path, 'r') as fp:
    #     cfg = json.load(fp)
    # cfg['root'] = '/home/dtrimina/Desktop/lxy/database/irseg'
    # dataset = IRSeg(cfg, mode='train', do_aug=True)
    # print(len(dataset))
    # from toolbox.utils import ClassWeight
    #
    # train_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg['ims_per_gpu'], shuffle=True,
    #                                            num_workers=cfg['num_workers'], pin_memory=True)
    # classweight = ClassWeight('enet')  # enet, median_freq_balancing
    # class_weight = classweight.get_weight(train_loader, 2)
    # class_weight = torch.from_numpy(class_weight).float()
    # # class_weight[cfg['id_unlabel']] = 0
    #
    # print(class_weight)
