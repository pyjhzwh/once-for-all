#! /bin/bash
#CUDA_VISIBLE_DEVICES=0,2,3 horovodrun -np 3 python3 train_ofa_resnet50.py --task expand --phase 1
#CUDA_VISIBLE_DEVICES=1,0,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task expand --phase 2
CUDA_VISIBLE_DEVICES=1,0,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task width --phase 1
#CUDA_VISIBLE_DEVICES=1,0,2 horovodrun -np 3 python3 train_ofa_resnet50.py --task width --phase 2
