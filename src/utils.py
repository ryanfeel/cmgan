import torch
import torch.nn as nn


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)


def power_compress_with_low_f_delete(spec, delete_f_bin=10):
    mag = torch.abs(spec)
    phase = torch.angle(spec)

    temp_mag = mag.clone()
    temp_phase = phase.clone()

    mag[:, :delete_f_bin, :] = 0
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1), temp_mag, temp_phase

def power_uncompress_with_low_f_delete(real, imag, temp_mag, temp_phase, delete_f_bin=10):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    
    mag = mag ** (1.0 / 0.3)
    
    '''
    # Frequency 차원에서 평균 계산
    mean_mag = torch.mean(mag[:, :, delete_f_bin:, :], dim=2, keepdim=True)  # [1, 1, 1, 1201]
    # mean_phase = torch.mean(phase[:, :, delete_f_bin:, :], dim=2, keepdim=True)

    # 정규화 (0에서 1 사이의 값으로 스케일링)
    min_val = torch.min(mean_mag)
    max_val = torch.max(mean_mag)
    norm_mag = (mean_mag - min_val) / (max_val - min_val)

    # min_val = torch.min(mean_phase)
    # max_val = torch.max(mean_phase)
    # norm_phase = (mean_phase - min_val) / (max_val - min_val)

    # 원래 텐서와 곱하기
    # print(temp_mag.size())
    temp_mag = temp_mag.unsqueeze(0) * norm_mag
    # temp_phase = temp_phase.unsqueeze(0) * norm_phase

    mag[:, :, :delete_f_bin, :] = temp_mag[:, :, :delete_f_bin, :]
    # phase[:, :, :delete_f_bin, :] = temp_phase.unsqueeze(0)[:, :, :delete_f_bin, :]
    
    '''

    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.complex(real_compress, imag_compress)

def power_compress(spec):
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag**0.3
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    return torch.stack([real_compress, imag_compress], 1)

def power_uncompress(real, imag):
    spec = torch.complex(real, imag)
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    mag = mag ** (1.0 / 0.3)
    real_compress = mag * torch.cos(phase)
    imag_compress = mag * torch.sin(phase)
    # return torch.stack([real_compress, imag_compress], -1)
    return torch.complex(real_compress, imag_compress)


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
