import torch
from torch import nn
from torchvision.models.vgg import vgg16, VGG16_Weights

import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

from torchvision import transforms


import numpy as np
import torch

def lmk68_symmetry_loss(landmarks):
    """
    Computes symmetry loss for 68 facial landmarks.

    Args:
        landmarks (torch.Tensor): Tensor of shape (68, 3).

    Returns:
        torch.Tensor: Symmetry loss value (scalar).
    """
    # Define the symmetric landmark pairs
    # Format: (index1, index2) where landmarks[index1] is symmetric to landmarks[index2]
    symmetric_pairs = [
        (0, 16), (1, 15), (2, 14), (3, 13), (4, 12), (5, 11), (6, 10), (7, 9),  # Jawline
        # (17, 26), (18, 25), (19, 24), (20, 23), (21, 22),  # Eyebrows
        # (36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46),  # Eyes
        # (31, 35), (32, 34),  # Nose
        # (48, 54), (49, 53), (50, 52), (51, 51), (58, 56), (59, 55),  # Mouth
    ]
    
    # Find the vertical center (axis of symmetry)
    x_center = landmarks[:, 0].mean()  # Average x-coordinate of all landmarks

    # Compute symmetry loss
    loss = 0
    for idx1, idx2 in symmetric_pairs:
        point1 = landmarks[idx1]
        point2 = landmarks[idx2]

        # Reflect point2 across the vertical axis
        reflected_point2 = point2.clone()
        reflected_point2[0] = 2 * x_center - reflected_point2[0]
        
        # Compute the L2 distance (squared) between the points
        loss += torch.sum((point1 - reflected_point2).abs())

    # Normalize loss by the number of pairs
    loss /= len(symmetric_pairs)
    return loss


def joint_prior_energy(frame_idx):
    """
    Regularizes the joints of the flame head model towards neutral joint locations
    """
    # poses = [
    #     ("neck", self.neck_pose[[frame_idx], :]),
    #     ("jaw", self.jaw_pose[[frame_idx], :]),
    #     ("eyes", self.eyes_pose[[frame_idx], :3]),
    #     ("eyes", self.eyes_pose[[frame_idx], 3:]),
    # ]

    # Joints should are regularized towards neural
    E_joint_prior = 0
    for name, pose in poses:
        # L2 regularization for each joint
        rotmats = batch_rodrigues(torch.cat([torch.zeros_like(pose), pose], dim=0))
        diff = ((rotmats[[0]] - rotmats[1:]) ** 2).mean()

        # Additional regularization for physical plausibility
        if name == 'jaw':
            # penalize negative rotation along x axis of jaw 
            diff += F.relu(-pose[:, 0]).mean() * 10

            # penalize rotation along y and z axis of jaw
            diff += (pose[:, 1:] ** 2).mean() * 3
        elif name == 'eyes':
            # penalize the difference between the two eyes
            diff += ((self.eyes_pose[[frame_idx], :3] - self.eyes_pose[[frame_idx], 3:]) ** 2).mean()

        E_joint_prior += diff * self.cfg.w[f"prior_{name}"]
    return E_joint_prior
    
    
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

        loss_network_1 = nn.Sequential(*list(vgg.features)[:4]).eval()
        for param in loss_network_1.parameters():
            param.requires_grad = False
        self.loss_network_1 = loss_network_1

        loss_network_2 = nn.Sequential(*list(vgg.features)[:9]).eval()
        for param in loss_network_2.parameters():
            param.requires_grad = False
        self.loss_network_2 = loss_network_2

        loss_network_3 = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in loss_network_3.parameters():
            param.requires_grad = False
        self.loss_network_3 = loss_network_3

        loss_network_4 = nn.Sequential(*list(vgg.features)[:23]).eval()
        for param in loss_network_4.parameters():
            param.requires_grad = False
        self.loss_network_4 = loss_network_4

        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        self.transform = transforms.Compose([
            transforms.Resize(size=(256, 256), antialias=True), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, input, target):

        input = self.transform(input[:, 0:3, :, :])
        target = self.transform(target[:, 0:3, :, :])

        perception_loss = 0.0
        perception_loss += self.l2_loss(self.loss_network_1(input), self.loss_network_1(target))
        # perception_loss += self.l2_loss(self.loss_network_2(input), self.loss_network_2(target))
        # perception_loss += self.l2_loss(self.loss_network_3(input), self.loss_network_3(target))
        # perception_loss += self.l2_loss(self.loss_network_4(input), self.loss_network_4(target))
        
        return perception_loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


if __name__ == '__main__':
    M = PerceptualLoss()
    img = torch.randn(1, 3, 512, 512)
    target = torch.randn(1, 3, 512, 512)
    print(M(img, target))