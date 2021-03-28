# LFSCA-UNet  
eg:  
　　cd ./Pytorch-UNet  
　　python train.py -e=5 -b=4 -l=1e-5 -v=0.1 -d=5 --model_type=unet --split_seed=12  
　　python test.py --model_type=unet --model=./checkpoints/Mar19_15-27-52_afed47d96e79_MT_unet_SS_12_LR_1e-05_BS_4/CP_epochs5.pth  
  
  
all masks and images are 256*256 .npy file  
reference: milesial/Pytorch-UNet
