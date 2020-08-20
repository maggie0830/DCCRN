import torch
import numpy as np
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
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1, bias=True, dropout=0, bidirectional=False):
        super().__init__()
        self.batch_size = batch_size
        self.lstm_re = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)
        self.lstm_im = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias,
                               dropout=dropout, bidirectional=bidirectional)
        self.h_real = torch.randn(num_layers, batch_size, hidden_size)
        self.h_imag = torch.randn(num_layers, batch_size, hidden_size)
        self.c_real = torch.randn(num_layers, batch_size, hidden_size)
        self.c_imag = torch.randn(num_layers, batch_size, hidden_size)

    def forward(self, x):
        real = self.lstm_re(x[..., 0], (self.h_real, self.c_real))[0] - \
               self.lstm_im(x[..., 1], (self.h_imag, self.c_imag))[0]
        imaginary = self.lstm_im(x[..., 0], (self.h_imag, self.c_imag))[0] + self.lstm_re(x[..., 1], (
            self.h_real, self.c_real))[0]
        output = torch.stack((real, imaginary), dim=-1)
        return output


class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__()
        self.bn_re = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats)
        self.bn_im = nn.BatchNorm2d(num_features=num_features, momentum=momentum, affine=affine, eps=eps,
                                    track_running_stats=track_running_stats)

    def forward(self, x):
        real = self.bn_re(x[..., 0])
        imag = self.bn_im(x[..., 1])
        output = torch.stack((real, imag), dim=-1)
        return output


class ComplexBatchNormal(nn.Module):

    def __init__(self, bs, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.gamma_rr = nn.Parameter(torch.randn(bs), requires_grad=True)
        self.gamma_ri = nn.Parameter(torch.randn(bs), requires_grad=True)
        self.gamma_ii = nn.Parameter(torch.randn(bs), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(bs), requires_grad=True)
        self.epsilon = 1e-5
        self.running_mean_real = 0
        self.running_mean_imag = 0
        self.Vrr = 0
        self.Vri = 0
        self.Vii = 0

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

            Vrr = torch.mean(real_centred * real_centred, 0) + self.epsilon  # [c,h,w]
            Vii = torch.mean(imag_centred * imag_centred, 0) + self.epsilon
            Vri = torch.mean(real_centred * imag_centred, 0)
            if self.running_mean + self.running_mean_imag + self.Vrr == 0:
                self.running_mean_real += mu_real
                self.running_mean_imag += mu_imag
                self.Vrr += Vrr
                self.Vri += Vri
                self.Vii += Vii
            else:
                self.running_mean_real = self.momentum * self.running_mean_real + (1 - self.momentum) * mu_real
                self.running_mean_imag = self.momentum * self.running_mean_imag + (1 - self.momentum) * mu_imag
                self.Vrr = self.momentum * self.Vrr + (1 - self.momentum) * Vrr
                self.Vri = self.momentum * self.Vri + (1 - self.momentum) * Vri
                self.Vii = self.momentum * self.Vii + (1 - self.momentum) * Vii
            return self.cbn(real_centred, imag_centred, Vrr, Vii, Vri, B, C, H, W)
        else:
            broadcast_mu_real = self.running_mean_real.repeat(B, 1, 1, 1)
            broadcast_mu_imag = self.running_mean_imag.repeat(B, 1, 1, 1)
            real_centred = real - broadcast_mu_real
            imag_centred = imaginary - broadcast_mu_imag
            self.cbn(real_centred, imag_centred, self.Vrr, self.Vii, self.Vri, B, C, H, W)

    def cbn(self, real_centred, imag_centred, Vrr, Vii, Vri, B, C, H, W):
        tau = Vrr + Vii
        delta = (Vrr * Vii) - (Vri ** 2)
        s = np.sqrt(delta)
        t = np.sqrt(tau + 2 * s)
        inverse_st = 1.0 / (s * t)

        Wrr = ((Vii + s) * inverse_st).repeat(B, 1, 1, 1)
        Wii = ((Vrr + s) * inverse_st).repeat(B, 1, 1, 1)
        Wri = (-Vri * inverse_st).repeat(B, 1, 1, 1)

        n_real = Wrr * real_centred + Wri * imag_centred
        n_imag = Wii * imag_centred + Wri * real_centred

        broadcast_gamma_rr = self.gamma_rr.view(B, 1, 1, 1).repeat(1, C, H, W)
        broadcast_gamma_ri = self.gamma_ri.view(B, 1, 1, 1).repeat(1, C, H, W)
        broadcast_gamma_ii = self.gamma_ii.view(B, 1, 1, 1).repeat(1, C, H, W)
        broadcast_beta = self.beta.view(B, 1, 1, 1).repeat(1, C, H, W)

        bn_real = broadcast_gamma_rr * n_real + broadcast_gamma_ri * n_imag + broadcast_beta
        bn_imag = broadcast_gamma_ri * n_real + broadcast_gamma_ii * n_imag + broadcast_beta
        return torch.stack((bn_real, bn_imag), dim=-1)


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
