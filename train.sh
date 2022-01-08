#! /bin/bash
#CUDA_VISIBLE_DEVICES=0,1,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task expand --phase 1
#CUDA_VISIBLE_DEVICES=0,1,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task expand --phase 2
#CUDA_VISIBLE_DEVICES=0,1,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task width --phase 1
#CUDA_VISIBLE_DEVICES=0,1,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task width --phase 2

CUDA_VISIBLE_DEVICES=0,1,2 horovodrun -np 3 python3 train_ofa_net.py --task expand --phase 3
