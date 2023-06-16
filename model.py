import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pretrain import D_VECTOR
from collections import OrderedDict
from data_loader import get_loader


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)

class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, num_speakers=10, repeat_num=6):
        super(Generator, self).__init__()
        c_dim = num_speakers
        layers = []
        layers.append(nn.Conv2d(1+c_dim, conv_dim, kernel_size=(3, 9), padding=(1, 4), bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck layers.
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        return self.main(x)

class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, input_size=(80, 256), conv_dim=64, repeat_num=5, num_speakers=10):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size_0 = int(input_size[0] / np.power(2, repeat_num)) # 1
        kernel_size_1 = int(input_size[1] / np.power(2, repeat_num)) # 8
        self.main = nn.Sequential(*layers)
        self.conv_dis = nn.Conv2d(curr_dim, 1, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False) # padding should be 0
        self.conv_clf_spks = nn.Conv2d(curr_dim, num_speakers, kernel_size=(kernel_size_0, kernel_size_1), stride=1, padding=0, bias=False)  # for num_speaker
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv_dis(h)
        out_cls_spks = self.conv_clf_spks(h)


        # speaker embedding vector
        C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).cuda(1)
        c_checkpoint = torch.load('3000000-BL.ckpt')
        new_state_dict = OrderedDict()  # 순서까지 인식하는 Dictionary 입력 클래스
        for key, val in c_checkpoint['model_b'].items():  # 앞에 module. 을 뺀 새 Dictionary
            new_key = key[7:]
            new_state_dict[new_key] = val
        C.load_state_dict(new_state_dict)

        x = np.squeeze(x) # (16, 1, 80, 256) -> (16, 80, 256)
        len_crop = 128
        out_spk_vec = torch.tensor([]).cuda(1)
        for i in range(x.shape[0]):
            tmp = torch.transpose(x[i], 0, 1)
            left = np.random.randint(0, tmp.shape[0] - len_crop)
            melsp = torch.tensor(tmp[np.newaxis, left:left + len_crop, :]).cuda(1)
            # 기존 (x,80)의 음성데이터를 torch size가 [1,128,80]인 고정 melsp로 변환
            emb = C(melsp.float())
            emb.squeeze()
            out_spk_vec = torch.cat((out_spk_vec, emb), dim=0)

        return out_src, out_cls_spks.view(out_cls_spks.size(0), out_cls_spks.size(1)), out_spk_vec

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    train_loader = get_loader('/scratch/sxliu/data_exp/VCTK-Corpus-22.05k/mc/train', 16, 'train', num_workers=1)
    data_iter = iter(train_loader)
    G = Generator().to(device)
    D = Discriminator().to(device)
    for i in range(10):
        mc_real, spk_label_org, acc_label_org, spk_acc_c_org = next(data_iter)
        mc_real.unsqueeze_(1) # (B, D, T) -> (B, 1, D, T) for conv2d
        mc_real = mc_real.to(device)                         # Input mc.
        spk_label_org = spk_label_org.to(device)             # Original spk labels.
        acc_label_org = acc_label_org.to(device)             # Original acc labels.
        spk_acc_c_org = spk_acc_c_org.to(device)             # Original spk acc conditioning.
        mc_fake = G(mc_real, spk_acc_c_org)
        print(mc_fake.size())
        out_src, out_cls_spks, out_cls_emos = D(mc_fake)



