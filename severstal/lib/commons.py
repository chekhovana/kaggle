import numpy as np
import pandas as pd
import os, cv2
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import CropNonEmptyMaskIfExists
from albumentations.augmentations.transforms import GaussianBlur, CLAHE
from albumentations import Flip, Normalize, Compose, OneOf, Sharpen
from albumentations.augmentations.geometric.resize import Resize

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler, RandomSampler

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
# mean = [0.3439]
# std = [0.0383]

# class SegformerAdapter(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.model = SegformerForSemanticSegmentation.from_pretrained(*args, **kwargs)

#     def forward(self, img):
#         logits = self.model(img).logits
#         upsampled_logits = nn.functional.interpolate(logits, size=img.shape[-2:],
#                                                      mode="bilinear",
#                                                      align_corners=False)
#         return upsampled_logits

# num_classes = config['dataset']['num_classes']
# id2label = {i: f'class_{i}' for i in range(num_classes)}
# label2id = {v: k for k, v in id2label.items()}

# model = SegformerAdapter("nvidia/mit-b5", ignore_mismatched_sizes=True,
#                          num_labels=num_classes, id2label=id2label, label2id=label2id,
#                          reshape_last_stage=True)


def make_mask(df_row, img_size):
    labels = df_row[:4]
    # mask = np.zeros((img_size['height'], img_size['width']), dtype=np.float32)
    mask = np.zeros(img_size['width'] * img_size['height'],
                    dtype=np.uint8)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            for pos, le in zip(positions, length):
                pos -= 1
                mask[pos : (pos + le)] = idx + 1

    mask = mask.reshape(img_size['height'], img_size['width'], order='F')
    return mask


class MaskConverter:
    def __init__(self, binary_mode=False, reduce_zero_label=False, without_background=True, **kwargs):
        self.binary_mode = binary_mode
        self.reduce_zero_label = reduce_zero_label
        self.without_background = without_background
        if self.binary_mode:
            self.channel = kwargs['channel']

    def __call__(self, mask):
        mask = mask.to(torch.int64)
        if self.binary_mode:
            if self.channel is not None:
                mask = (mask == self.channel).to(torch.int64)
            else:
                mask[mask > 1] = 1
            mask_onehot = mask.unsqueeze(0).float()
            # mask_onehot = F.one_hot(mask, num_classes=2)
            # mask_onehot = mask_onehot.permute(2, 0, 1).to(torch.float64)
            return mask, mask_onehot

        mask_onehot = F.one_hot(mask, num_classes=5).permute(2, 0, 1)
        mask_onehot = mask_onehot.to(torch.float64)

        if self.without_background:
            mask_onehot = mask_onehot[1:]

        if self.reduce_zero_label:
            mask = mask - 1
            mask[mask == -1] = 255

        return mask, mask_onehot


class SteelSegmentationDataset(Dataset):
    @staticmethod
    def get_train_transforms(crop_size):
        return Compose([
            CropNonEmptyMaskIfExists(crop_size['height'], crop_size['width']),
            OneOf([
                CLAHE(p=0.5),  # modified source to get this to work
                GaussianBlur(blur_limit=(3, 3), p=0.3),
                Sharpen(alpha=(0.2, 0.3), p=0.3),
            ], p=1),
            # Resize(crop_size['height'] * 4, crop_size['width'] * 4),
            Flip(p=0.5),
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    @staticmethod
    def get_valid_transforms():
        return Compose([
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __init__(self, df, img_folder, is_train, img_size, **kwargs):
        self.df = df
        self.img_folder = img_folder
        if is_train:
            self.transforms = SteelSegmentationDataset.get_train_transforms(kwargs['crop_size'])
        else:
            self.transforms = SteelSegmentationDataset.get_valid_transforms()
        self.img_size = img_size
        self.mask_converter = MaskConverter(**kwargs['mask'])

    def __len__(self):
        return self.df.shape[0]

    def remove_background(self, image, mask):
        mask = (mask > 0).astype(image.dtype)
        mask = np.array([mask] * 3).transpose(1, 2, 0)
        image = image * mask
        return image

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = df_row.name
        image_path = os.path.join(self.img_folder, image_id)
        image = cv2.imread(image_path)
        mask = make_mask(df_row, self.img_size)
        if self.mask_converter.reduce_zero_label:
            image = self.remove_background(image, mask)
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        mask, mask_onehot = self.mask_converter(mask)
        data = {'image': image, 'mask': mask, 'mask_onehot': mask_onehot}
        return data


class SteelClassificationDataset(Dataset):
    @staticmethod
    def get_train_transforms():
        return Compose([
            # OneOf([
            #     CLAHE(p=0.5),  # modified source to get this to work
            #     GaussianBlur(blur_limit=(3, 3), p=0.3),
            #     Sharpen(alpha=(0.2, 0.3), p=0.3),
            # ], p=1),
            # Resize(256, 256, always_apply=True, p=1),
            Flip(p=0.5),
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    @staticmethod
    def get_valid_transforms():
        return Compose([
            Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    def __init__(self, df, img_folder, is_train, img_size, **kwargs):
        self.df = df
        self.img_folder = img_folder
        if is_train:
            self.transforms = SteelClassificationDataset.get_train_transforms()
        else:
            self.transforms = SteelClassificationDataset.get_valid_transforms()
        self.img_size = img_size

    def __len__(self):
        return self.df.shape[0]


    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        image_id = df_row.name
        image_path = os.path.join(self.img_folder, image_id)
        image = cv2.imread(image_path)
        augmented = self.transforms(image=image)
        image = augmented['image']
        mask = df_row[['1', '2', '3', '4']].notna().values.tolist()
        mask = torch.tensor(mask).float()
        data = {'image': image, 'mask': mask}
        return data


def get_split(df, split_fname):
    with open(split_fname) as f:
        idx_list = [l.replace('\n', '.jpg') for l in f.readlines()]
    return df[df.index.isin(idx_list)]


def create_loaders(folder=None, split=None, **kwargs):
    img_folder = os.path.join(folder, 'images')
    df = pd.read_csv(os.path.join(folder, 'train_converted.csv'), index_col='ImageId')
    # df['ClassId'] = df['ClassId'].astype(int)
    # df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    # df['defects'] = df.count(axis=1)
    train_split = os.path.join(folder, split, 'train.txt')
    val_split = os.path.join(folder, split, 'val.txt')
    train_df, val_df = get_split(df, train_split), get_split(df, val_split)
    ds_type = kwargs['type']
    assert ds_type in ('classification', 'segmentation')
    batch_size = kwargs['batch_size']
    if ds_type == 'segmentation':
        train_dataset = SteelSegmentationDataset(train_df, img_folder, is_train=True, **kwargs)
        val_dataset = SteelSegmentationDataset(val_df, img_folder, is_train=False, **kwargs)
    else:
        train_dataset = SteelClassificationDataset(train_df, img_folder, is_train=True, **kwargs)
        val_dataset = SteelClassificationDataset(val_df, img_folder, is_train=False, **kwargs)
    # train_sampler = WeightedRandomSampler(torch.tensor(train_df['weight'].values.tolist()),
    #     num_samples=kwargs['size']['train'])
    # val_sampler = WeightedRandomSampler(torch.tensor(val_df['weight'].values.tolist()),
    #     num_samples=kwargs['size']['val'])
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size['train'], shuffle=True),
        'valid': DataLoader(val_dataset, batch_size=batch_size['valid'], shuffle=False)
    }
    return loaders

from typing import Optional
from functools import partial
from catalyst.metrics import RegionBasedMetric

def apply_threshold(outputs, threshold):
    logits = outputs * (outputs > threshold)
    num_classes = outputs.shape[1]
    logits = logits.permute(1, 0, 2, 3)
    index = logits.argmax(dim=0).unsqueeze(0)
    logits = logits.gather(0, index).squeeze(0)
    index = index + 1
    index = index * (logits > 0)
    index = index.squeeze(0)
    index = F.one_hot(index, num_classes=num_classes + 1)[..., 1:]
    index = index.permute(0, 3, 1, 2).float()
    return ((outputs * index) > threshold).float()

def dice_per_image(outputs, targets, threshold=None, softmax=False):
    sum_func = partial(torch.sum, dim=[2, 3])
    if softmax:
        outputs = F.one_hot(outputs.argmax(dim=1), num_classes=targets.shape[1])
        outputs = outputs.permute(0, 3, 1, 2)
    elif threshold is not None:
        outputs = apply_threshold(outputs, threshold)
    nominator = 2 * sum_func(outputs * targets)
    targets_sum = sum_func(targets)
    denominator = sum_func(outputs) + targets_sum
    eps = 1e-7 * (denominator == 0)
    res = (nominator + eps) / (denominator + eps)
    # res = res * (targets_sum > 0)
    res = res.mean(dim=0)
    return res

class DiceLossPerImage(nn.Module):
    def __init__(self):
        super(DiceLossPerImage, self).__init__()

    def forward(self, outputs, targets):
        return 1 - dice_per_image(outputs, targets).mean()

class DiceMetricPerImage(RegionBasedMetric):

    def __init__(
        self,
        metric_name: str = 'dice_pi',
        threshold: Optional[float] = 0.5,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
        softmax=False,
    ):
        super().__init__(compute_on_call, prefix, suffix)
        self.metric_name = metric_name
        self.threshold = threshold
        self._checked_params = False
        self._ddp_backend = None
        self.softmax = softmax
        self.buffer = None
        self.count = 0

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        res = dice_per_image(outputs, targets, threshold=self.threshold, softmax=self.softmax)
        if self.buffer is None:
            self.buffer = res
        else:
            self.buffer += res
        self.count += 1
        if not self._checked_params:
            self.class_names = [f"class_{idx:02d}" for idx in range(len(res))]
            self._checked_params = True
        return res

    def compute(self):
        per_class = self.buffer / self.count
        return per_class, 0, per_class.mean(), 0

    def reset(self):
        self.buffer = None
        self.count = 0

class DiceMetricPosNeg(RegionBasedMetric):

    def __init__(
        self,
        metric_name: str = 'dice_pi',
        threshold: Optional[float] = 0.5,
        compute_on_call: bool = True,
        prefix: Optional[str] = None,
        suffix: Optional[str] = None,
    ):
        super().__init__(compute_on_call, prefix, suffix)
        self.metric_name = metric_name
        self.threshold = threshold
        self._checked_params = False
        self._ddp_backend = None
        self.first_pass = True

    def init(self, shape):
        self.first_pass = False
        self.count_pos = torch.zeros(shape).cuda()
        self.count_neg = torch.zeros(shape).cuda()
        self.count_all = torch.zeros(shape).cuda()
        self.dice_pos = torch.zeros(shape).cuda()
        self.dice_neg = torch.zeros(shape).cuda()
        self.dice_all = torch.zeros(shape).cuda()

    def get_masked_dice(self, nominator, denominator, mask):
        mask = mask.cuda()
        nominator = nominator# * mask
        denominator = denominator# * mask
        eps = 1e-7 * (denominator == 0)
        dice = (nominator + eps) / (denominator + eps)
        return dice * mask

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        sum_func = partial(torch.sum, dim=[2, 3])
        nominator = 2 * sum_func(outputs * targets)
        targets_sum = sum_func(targets)
        outputs_sum = sum_func(outputs)
        denominator = outputs_sum + targets_sum

        if not self._checked_params:
            self.class_names = [f"class_{idx:02d}" for idx in range(targets.shape[1])]
            self._checked_params = True

        if self.first_pass:
            self.init(nominator.shape)

        mask_pos = targets_sum != 0
        self.dice_pos += self.get_masked_dice(nominator, denominator, mask_pos)
        self.count_pos += mask_pos

        mask_neg = targets_sum == 0
        self.dice_neg += self.get_masked_dice(nominator, denominator, mask_neg)
        self.count_neg += mask_neg

        mask_all = torch.ones(targets_sum.shape).cuda()
        self.dice_all += self.get_masked_dice(nominator, denominator, mask_all)
        self.count_all += mask_all
        print('debug dice_all shape', self.dice_all.shape)
        return self.dice_all.mean()

    def compute(self):
        dice_pos_mean = self.dice_pos / self.count_pos
        dice_neg_mean = self.dice_neg / self.count_neg
        dice_avg_mean = (dice_pos_mean + dice_neg_mean) / 2
        return dice_avg_mean.mean(dim=0), dice_avg_mean.mean(), dice_pos_mean.mean(), dice_neg_mean.mean()

    def update_key_value(self, outputs: torch.Tensor, targets: torch.Tensor):
        # import traceback
        # traceback.print_stack()
        self.update(outputs, targets)
        return self.compute_key_value()

    def compute_key_value(self):
        avg_per_class, avg_mean, pos_mean, neg_mean = self.compute()
        print(self.class_names)
        metrics = {}
        print('debug avg per class', avg_per_class.shape)
        print('debug class names', self.class_names)
        for class_idx, value in enumerate(avg_per_class):

            metrics[
                f"{self.prefix}{self.metric_name}{self.suffix}/{self.class_names[class_idx]}"
            ] = value

        metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_neg"] = neg_mean
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}"] = avg_mean
        metrics[f"{self.prefix}{self.metric_name}{self.suffix}/_pos"] = pos_mean
        return metrics

    def reset(self):
        self.first_pass = True

if __name__ == '__main__':
    import yaml
    with open('config/classification.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    loaders = create_loaders(**config['dataset'])
    ds = loaders['train'].dataset
    print(ds[0]['mask'])

