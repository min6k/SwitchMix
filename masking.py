import random
import warnings

import kornia
import numpy as np
import torch
from einops import repeat
from torch import nn, Tensor
from torch.nn import functional as F

warnings.filterwarnings("ignore", category=DeprecationWarning)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


def strong_transform(param, data):
    data = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data)
    data = gaussian_blur(blur=param['blur'], data=data)
    return data


def denorm(img, mean, std):
    return img.mul(std).add(mean)


def renorm(img, mean, std):
    return img.sub(mean).div(std)


def color_jitter(color_jitter, mean, std, data, s=.25, p=.2):
    # s is the strength of colorjitter
    if color_jitter > p:
        mean = torch.as_tensor(mean, device=data.device)
        mean = repeat(mean, 'C -> B C 1 1', B=data.shape[0], C=3)
        std = torch.as_tensor(std, device=data.device)
        std = repeat(std, 'C -> B C 1 1', B=data.shape[0], C=3)
        if isinstance(s, dict):
            seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
        else:
            seq = nn.Sequential(
                kornia.augmentation.ColorJitter(
                    brightness=s, contrast=s, saturation=s, hue=s))
        data = denorm(data, mean, std)
        data = seq(data)
        data = renorm(data, mean, std)
    return data


def gaussian_blur(blur, data):
    if blur > 0.5:
        sigma = np.random.uniform(0.15, 1.15)
        kernel_size_y = int(
            np.floor(
                np.ceil(0.1 * data.shape[2]) - 0.5 +
                np.ceil(0.1 * data.shape[2]) % 2))
        kernel_size_x = int(
            np.floor(
                np.ceil(0.1 * data.shape[3]) - 0.5 +
                np.ceil(0.1 * data.shape[3]) % 2))
        kernel_size = (kernel_size_y, kernel_size_x)
        seq = nn.Sequential(
            kornia.filters.GaussianBlur2d(
                kernel_size=kernel_size, sigma=(sigma, sigma)))
        data = seq(data)
    return data

class Masking(nn.Module):
    def __init__(self, block_size=64, ratio=0.5, color_jitter_s=0.2, color_jitter_p=0.2, blur=True, mean= [102.9801, 115.9465, 122.7717], std= [1., 1., 1.]):
        super(Masking, self).__init__()

        self.block_size = block_size
        self.ratio = ratio

        self.augmentation_params = None
        if (color_jitter_p > 0 and color_jitter_s > 0) or blur:
            print('[Masking] Use color augmentation.')
            self.augmentation_params = {
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': color_jitter_s,
                'color_jitter_p': color_jitter_p,
                'blur': random.uniform(0, 1) if blur else 0,
                'mean': mean,
                'std': std
            }

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        if self.augmentation_params is not None:
            img = strong_transform(self.augmentation_params, data=img.clone())

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()
        input_mask = resize(input_mask, size=(H, W))
        masked_img = img * input_mask

        return masked_img

class Masking_only(nn.Module):
    def __init__(self, block_size=32, ratio=0.5):
        super(Masking_only, self).__init__()  # ← 수정됨
        self.block_size = block_size
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()

        input_mask = resize(input_mask, size=(H, W))

        if input_mask.shape[1] == 1:
            input_mask = input_mask.expand(-1, img.shape[1], -1, -1)

        masked_img = img * input_mask
        return masked_img


class Masking_only_2(nn.Module):
    def __init__(self, block_size=64, ratio=0.7):
        super(Masking_only_2, self).__init__()  # ← 수정됨
        self.block_size = block_size
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, _, H, W = img.shape

        mshape = B, 1, round(H / self.block_size), round(W / self.block_size)
        input_mask = torch.rand(mshape, device=img.device)
        input_mask = (input_mask > self.ratio).float()

        input_mask = resize(input_mask, size=(H, W))

        if input_mask.shape[1] == 1:
            input_mask = input_mask.expand(-1, img.shape[1], -1, -1)

        masked_img = img * input_mask
        return masked_img

class DualMasking(nn.Module):
    def __init__(self, block_size=32, ratio=0.5):
        super(DualMasking, self).__init__()
        self.block_size = block_size
        self.ratio = ratio

    @torch.no_grad()
    def forward(self, img: Tensor):
        img = img.clone()
        B, C, H, W = img.shape

        # 작은 해상도로 마스크 생성
        mshape = (B, 1, round(H / self.block_size), round(W / self.block_size))
        mask = torch.rand(mshape, device=img.device)
        mask = (mask > self.ratio).float()  # 1: 보이는 부분, 0: 가릴 부분

        # 마스크 사이즈 원래 이미지에 맞추기
        mask = resize(mask, size=(H, W))

        # 채널 수 일치시키기
        if mask.shape[1] == 1:
            mask = mask.expand(-1, C, -1, -1)

        # 첫 번째: 보이는 부분(=mask)만 보이게
        masked_img_1 = img * mask

        # 두 번째: 나머지 부분(1-mask)만 보이게
        masked_img_2 = img * (1 - mask)

        return masked_img_1, masked_img_2