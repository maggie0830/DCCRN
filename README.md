# DCCRN
implementation of "DCCRN-Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement"
## how to run
```text
numpy==1.19.0
librosa==0.8.0
torch==1.6.0
torchaudio-contrib==0.1 (git+https://github.com/keunwoochoi/torchaudio-contrib@61fc6a804c941dec3cf8a06478704d19fc5e415a)
```
* change the "dns_home" of train.py to the dir of dns-datas
```text
-dns_datas/
    -clean/
    -noise/
    -noisy/
```
## explain
* batch_size and load_batch
```text
Each speech (16000*30) is divided into 800*600 (according to the paper, each frame is 37.5ms->37.5*16000/1000=600).
Load_batch: Number of sounds loaded into memory. 
Batch_size: Number of inputs to calculate (selected from "load_Batch *800").Ensure: 800%batch_size==0;Batch_size =800 or 400 or 200 or 100 or 50
```
