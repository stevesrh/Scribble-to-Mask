#!/usr/bin/env bash
echo Which PYTHON: `which python`
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=4 python -m torch.distributed.launch --nproc_per_node=2  \
     --master_port 9842  \
     --nproc_per_node=2 \
     train.py \
     --id 's2m' \
     --load_deeplab /home/srh/Scribble-to-Mask/best_deeplabv3plus_resnet50_voc_os16.pth \
     --static_root /media/liuyang/TOSHIBA2/scribble2mask_data/static  \
     --lvis_root  /media/liuyang/TOSHIBA2/scribble2mask_data/LVIS  \
     --iterations 10000 \
     --batch_size 12




