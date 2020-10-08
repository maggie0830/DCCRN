# coding: utf-8
# Author：WangTianRui
# Date ：2020/9/30 10:55


from complex_progress import *
import torchaudio_contrib as audio_nn
from utils import *


class STFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length = n_fft, hop_length
        self.stft = audio_nn.STFT(fft_length=self.n_fft, hop_length=self.hop_length, win_length=win_length)

    def forward(self, signal):
        with torch.no_grad():
            x = self.stft(signal)
            mag, phase = audio_nn.magphase(x, power=1.)

        mix = torch.stack((mag, phase), dim=-1)
        return mix.unsqueeze(1)


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft, self.hop_length, self.win_length = n_fft, hop_length, win_length

    def forward(self, x):
        B, C, F, T, D = x.shape
        x = x.view(B, F, T, D)
        x_istft = istft(x, hop_length=self.hop_length, length=600)
        return x_istft.view(B, C, -1)


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        if padding is None:
            padding = [int((i - 1) / 2) for i in kernel_size]  # same
            # padding
        self.conv = ComplexConv2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                  stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train):
        x = self.conv(x)

        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, chw, padding=None):
        super().__init__()
        self.transconv = ComplexConvTranspose2d(in_channel=in_channel, out_channel=out_channel, kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        self.bn = ComplexBatchNormal(chw[0], chw[1], chw[2])
        self.prelu = nn.PReLU()

    def forward(self, x, train=True):
        x = self.transconv(x)
        x = self.bn(x, train)
        x = self.prelu(x)
        return x


class DCCRN(nn.Module):
    def __init__(self, net_params, device, batch_size=36):
        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.encoders = []
        self.lstms = []
        self.dense = ComplexDense(net_params["dense"][0], net_params["dense"][1])
        self.decoders = []
        # init encoders
        en_channels = net_params["encoder_channels"]
        en_ker_size = net_params["encoder_kernel_sizes"]
        en_strides = net_params["encoder_strides"]
        en_padding = net_params["encoder_paddings"]
        encoder_chw = net_params["encoder_chw"]
        decoder_chw = net_params["decoder_chw"]
        for index in range(len(en_channels) - 1):
            model = Encoder(
                in_channel=en_channels[index], out_channel=en_channels[index + 1],
                kernel_size=en_ker_size[index], stride=en_strides[index], padding=en_padding[index],
                chw=encoder_chw[index]
            )
            self.add_module("encoder{%d}" % index, model)
            self.encoders.append(model)
        # init lstm
        lstm_dims = net_params["lstm_dim"]
        for index in range(len(net_params["lstm_dim"]) - 1):
            model = ComplexLSTM(input_size=lstm_dims[index], hidden_size=lstm_dims[index + 1],
                                num_layers=net_params["lstm_layer_num"], device=self.device)
            self.lstms.append(model)
            self.add_module("lstm{%d}" % index, model)
        # init decoder
        de_channels = net_params["decoder_channels"]
        de_ker_size = net_params["decoder_kernel_sizes"]
        de_strides = net_params["decoder_strides"]
        de_padding = net_params["decoder_paddings"]
        for index in range(len(de_channels) - 1):
            model = Decoder(
                in_channel=de_channels[index] + en_channels[len(self.encoders) - index],
                out_channel=de_channels[index + 1],
                kernel_size=de_ker_size[index], stride=de_strides[index], padding=de_padding[index],
                chw=decoder_chw[index]
            )
            self.add_module("decoder{%d}" % index, model)
            self.decoders.append(model)

        self.encoders = nn.ModuleList(self.encoders)
        self.lstms = nn.ModuleList(self.lstms)
        self.decoders = nn.ModuleList(self.decoders)
        self.linear = ComplexConv2d(in_channel=2, out_channel=1, kernel_size=1, stride=1)

    def forward(self, x, train=True):
        skiper = []
        for index, encoder in enumerate(self.encoders):
            skiper.append(x)
            x = encoder(x, train)
        B, C, F, T, D = x.size()
        lstm_ = x.reshape(B, -1, T, D)
        lstm_ = lstm_.permute(2, 0, 1, 3)
        for index, lstm in enumerate(self.lstms):
            lstm_ = lstm(lstm_)
        lstm_ = lstm_.permute(1, 0, 2, 3)
        lstm_out = lstm_.reshape(B * T, -1, D)
        dense_out = self.dense(lstm_out)
        dense_out = dense_out.reshape(B, T, C, F, D)
        p = dense_out.permute(0, 2, 3, 1, 4)
        for index, decoder in enumerate(self.decoders):
            p = decoder(p, train)
            p = torch.cat([p, skiper[len(skiper) - index - 1]], dim=1)
        mask = torch.tanh(self.linear(p))
        return mask


class DCCRN_(nn.Module):
    def __init__(self, n_fft, hop_len, net_params, batch_size, device, win_length):
        super().__init__()
        self.stft = STFT(n_fft, hop_len, win_length=win_length)
        self.DCCRN = DCCRN(net_params, device=device, batch_size=batch_size)
        self.istft = ISTFT(n_fft, hop_len, win_length=win_length)

    def forward(self, signal, train=True):
        stft = self.stft(signal)
        mask_predict = self.DCCRN(stft, train=train)
        predict = stft * mask_predict
        clean = self.istft(predict)
        return clean
