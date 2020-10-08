# coding: utf-8
# Author：WangTianRui
# Date ：2020/8/18 9:43

import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv_re = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)
        self.conv_im = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        real = self.conv_re(x[..., 0]) - self.conv_im(x[..., 1])
        imaginary = self.conv_re(x[..., 1]) + self.conv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, device, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.num_layer = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.lstm_re = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)
        self.lstm_im = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)

    def forward(self, x):
        batch_size = x.size(1)
        h_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        h_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        c_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        c_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        real_real, (h_real, c_real) = self.lstm_re(x[..., 0], (h_real, c_real))
        imag_imag, (h_imag, c_imag) = self.lstm_im(x[..., 1], (h_imag, c_imag))
        real = real_real - imag_imag
        h_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        h_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        c_real = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        c_imag = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device=self.device)
        imag_real, (h_real, c_real) = self.lstm_re(x[..., 1], (h_real, c_real))
        real_imag, (h_imag, c_imag) = self.lstm_im(x[..., 0], (h_imag, c_imag))
        imaginary = imag_real + real_imag
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexDense(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.linear_read = nn.Linear(in_channel, out_channel)
        self.linear_imag = nn.Linear(in_channel, out_channel)

    def forward(self, x):
        real = x[..., 0]
        imag = x[..., 1]
        real = self.linear_read(real)
        imag = self.linear_imag(imag)
        out = torch.stack((real, imag), dim=-1)
        return out


class ComplexBatchNormal(nn.Module):
    def __init__(self, C, H, W, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(C, H, W), requires_grad=True)
        self.epsilon = 1e-5
        self.running_mean_real = None
        self.running_mean_imag = None
        self.Vrr = None
        self.Vri = None
        self.Vii = None

    def forward(self, x, train=True):
        B, C, H, W, D = x.size()
        real = x[..., 0]
        imaginary = x[..., 1]
        if train:
            mu_real = torch.mean(real, dim=0)
            mu_imag = torch.mean(imaginary, dim=0)

            broadcast_mu_real = mu_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = mu_imag.repeat(B, 1, 1, 1)

            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag

            Vrr = torch.mean(real_centred * real_centred, 0) + self.epsilon
            Vii = torch.mean(imag_centred * imag_centred, 0) + self.epsilon
            Vri = torch.mean(real_centred * imag_centred, 0)
            if self.Vrr is None:
                self.running_mean_real = mu_real
                self.running_mean_imag = mu_imag
                self.Vrr = Vrr  # C,H,W
                self.Vri = Vri
                self.Vii = Vii
            else:
                # momentum
                self.running_mean_real = self.momentum * self.running_mean_real + (1 - self.momentum) * mu_real
                self.running_mean_imag = self.momentum * self.running_mean_imag + (1 - self.momentum) * mu_imag
                self.Vrr = self.momentum * self.Vrr + (1 - self.momentum) * Vrr
                self.Vri = self.momentum * self.Vri + (1 - self.momentum) * Vri
                self.Vii = self.momentum * self.Vii + (1 - self.momentum) * Vii
            return self.cbn(real_centred, imag_centred, Vrr, Vii, Vri, B)
        else:
            broadcast_mu_real = self.running_mean_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = self.running_mean_imag.repeat(B, 1, 1, 1)
            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag
            return self.cbn(real_centred, imag_centred, self.Vrr, self.Vii, self.Vri, B)

    def cbn(self, real_centred, imag_centred, Vrr, Vii, Vri, B):
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = torch.sqrt(delta)
        t = torch.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)

        Wrr = ((Vii + s) * inverse_st).repeat(B, 1, 1, 1)
        Wii = ((Vrr + s) * inverse_st).repeat(B, 1, 1, 1)
        Wri = (-Vri * inverse_st).repeat(B, 1, 1, 1)

        n_real = Wrr * real_centred + Wri * imag_centred
        n_imag = Wii * imag_centred + Wri * real_centred

        broadcast_gamma_rr = self.gamma_rr.repeat(B, 1, 1, 1)
        broadcast_gamma_ri = self.gamma_ri.repeat(B, 1, 1, 1)
        broadcast_gamma_ii = self.gamma_ii.repeat(B, 1, 1, 1)
        broadcast_beta = self.beta.repeat(B, 1, 1, 1)

        bn_real = broadcast_gamma_rr * n_real + broadcast_gamma_ri * n_imag + broadcast_beta
        bn_imag = broadcast_gamma_ri * n_real + broadcast_gamma_ii * n_imag + broadcast_beta
        return torch.stack((bn_real, bn_imag), dim=-1)


def init_get(kind):
    if kind == "sqrt_init":
        return sqrt_init
    else:
        return torch.zeros


def sqrt_init(shape):
    return (1 / torch.sqrt(torch.tensor(2))) * torch.ones(shape)


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, output_padding=0, dilation=1,
                 groups=1, bias=True):
        super().__init__()

        self.tconv_re = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)
        self.tconv_im = nn.ConvTranspose2d(in_channel, out_channel,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=output_padding,
                                           groups=groups,
                                           bias=bias,
                                           dilation=dilation)

    def forward(self, x):
        real = self.tconv_re(x[..., 0]) - self.tconv_im(x[..., 1])
        imaginary = self.tconv_re(x[..., 1]) + self.tconv_im(x[..., 0])
        output = torch.stack((real, imaginary), dim=-1)
        return output
