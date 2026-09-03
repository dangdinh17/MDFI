# MDFI
The official Pytorch implementation of "MDFI: A Multi-Domain Features Integration for Video Quality Enhancement from Versatile Video Coding"

The official code and dataset will be abvailable when the research is accepted.

# *MDFI: A Multi-Domain Features Integration for Video Quality Enhancement from Versatile Video Coding*


The *PyTorch* implementation for the [MDFI: A Multi-Domain Features Integration for Video Quality Enhancement from Versatile Video Coding](https://ieeexplore.ieee.org/document/11661787) which is accepted by [IEEE TCE].

Task: Video Quality Enhancement / Video Artifact Reduction.



## 1. Pre-request

### 1.1. Environment
Suppose that you have installed CUDA 11.0, then:
```bash
conda create -n mdfi python=3.8 -y  
conda activate mdfi
git clone --depth=1 https://github.com/dangdinh17/MDFI.git && cd MDFI/
python -m pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tqdm lmdb pyyaml opencv-python scikit-image thop
```

### 1.2. Dataset

Please check [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset).

### 1.3. Create LMDB
We now generate LMDB to speed up IO during training.
```bash
python create_lmdb_ovqe.py
```

## 2. Train

We utilize 1 NVIDIA Tesla V100 32GB for training.
```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_ovqe.yml
```

## 3. Test         
Pretrained models can be found here: [[GoogleDisk]]() and 

We utilize 1 NVIDIA Tesla V100 32GB for testing.
```bash
python test_mdfi.py
```

## Citation
If you find this project is useful for your research, please cite:
```bash
@ARTICLE{mdfi,
  author={NguyenQuang, Sang and Minh, Hieu Bui and BuiDinh, Dang and HoangVan, Xiem},
  journal={IEEE Transactions on Consumer Electronics}, 
  title={MDFI: A Multi-Domain Features Integration for Compressed Video Quality Enhancement}, 
  year={2026},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCE.2026.3725873}}

```

## Acknowledgements
This work is based on [STDF-Pytoch](https://github.com/RyanXingQL/STDF-PyTorch). Thank [RyanXingQL](https://github.com/RyanXingQL)  for sharing the codes.