# coding: utf-8
# Author：WangTianRui
# Date ：2020/9/29 21:47

import wav_loader as loader
import net_config as net_config
import pickle
from torch.utils.data import DataLoader
import module as model_cov_bn
from si_snr import *
import train_utils
import os

########################################################################
# Change the path to the path on your computer
dns_home = r"F:\Traindata\DNS-Challenge\make_data"  # dir of dns-datas
save_file = "./logs"  # model save
########################################################################

batch_size = 400  # calculate batch_size
load_batch = 100  # load batch_size(not calculate)
device = torch.device("cuda:0")  # device

lr = 0.001  # learning_rate
# load train and test name , train:test=4:1
if os.path.exists(r'./train_test_names.data'):
    train_test = pickle.load(open('./train_test_names.data', "rb"))
else:
    train_test = train_utils.get_train_test_name(dns_home)
train_noisy_names, train_clean_names, test_noisy_names, test_clean_names = \
    train_utils.get_all_names(train_test, dns_home=dns_home)

train_dataset = loader.WavDataset(train_noisy_names, train_clean_names, frame_dur=37.5)
test_dataset = loader.WavDataset(test_noisy_names, test_clean_names, frame_dur=37.5)
# dataloader
train_dataloader = DataLoader(train_dataset, batch_size=load_batch, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=load_batch, shuffle=True)

dccrn = model_cov_bn.DCCRN_(
    n_fft=512, hop_len=int(6.25 * 16000 / 1000), net_params=net_config.get_net_params(), batch_size=batch_size,
    device=device, win_length=int((25 * 16000 / 1000))).to(device)

optimizer = torch.optim.Adam(dccrn.parameters(), lr=lr)
criterion = SiSnr()
train_utils.train(model=dccrn, optimizer=optimizer, criterion=criterion, train_iter=train_dataloader,
                  test_iter=test_dataloader, max_epoch=500, device=device, batch_size=batch_size, log_path=save_file,
                  just_test=False)
