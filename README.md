# DCCRN
implementation of "DCCRN-Deep Complex Convolution Recurrent Network for Phase-Aware Speech Enhancement"
## how to run
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
Each speech (16,000 *30) is divided into 800*600 (according to the text, each frame is 37.5ms).
Load_batch: Number of sounds loaded into memory
Batch_size: Number of inputs to calculate (selected from "load_Batch *800")
```