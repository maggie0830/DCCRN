import torch
from complex_progress import *
import torchaudio_contrib as audio_nn
from utils import *


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.stft = audio_nn.STFT(fft_length=self.n_fft, hop_length=self.hop_length)

    def forward(self, signal):
        with torch.no_grad():
            x = self.stft(signal)
            mag, phase = audio_nn.magphase(x, power=1.)
        return mag, phase


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length

    def forward(self, x):
        B, C, F, T, D = x.shape
        x = x.view(B * C, F, T, D)
        x_istft = istft(x, hop_length=self.hop_length, win_length=self.n_fft)
        return x_istft.view(B, C, -1)


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bs, padding=None):
        super().__init__()
        if padding is None:
            padding = [(i - 1) // 2 for i in kernel_size]  # same
        self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        # self.bn = ComplexBatchNorm2d(out_channel)
        self.bn = ComplexBatchNormal(bs)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bs, padding=None):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        # self.bn = ComplexBatchNorm2d(out_channel)
        self.bn = ComplexBatchNormal(bs)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.transconv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class DCCRN(nn.Module):
    def __init__(self, net_params, input_channel=1, batch_size=36):
        super().__init__()
        self.encoders = []
        self.lstms = ComplexLSTM(input_size=36, hidden_size=128, batch_size=batch_size, num_layers=2)
        self.dense = nn.Linear(in_features=128, out_features=1280)
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        for index in range(len(en_channels - 1)):
            self.encoders.append(
                Encoder(
                    in_channel=en_channels[index], out_channel=en_channels[index + 1],
                    kernel_size=en_ker_size[index], stride=en_strides[index],
                    bs=batch_size, padding=en_padding[index]
                )
            )
        # init decoder



class DCCRN_(nn.Module):
    def __init__(self, x_shape, n_fft, hop_len):
        """
        B:BatchSize;C:channel;H:height;W:width;D:ComplexDim=2
        :param x_shape:
        """
        super().__init__()
        self.B, self.C, self.H, self.W, self.D = x_shape
        self.stft = STFT(n_fft, hop_len)
        self.istft = ISTFT(n_fft, hop_len)
