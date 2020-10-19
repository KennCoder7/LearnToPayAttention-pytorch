# Introduction
PyTorch implementation of ICLR 2018 paper [Learn To Pay Attention](https://openreview.net/forum?id=HyzbhfWRW).  
Including two attention method (dot product and parametrise) and visualization of attention map.  
CIFAR-10:  
![frog](https://github.com/KennCoder7/LearnToPayAttention/blob/main/frog.png)
![bird](https://github.com/KennCoder7/LearnToPayAttention/blob/main/bird.png)
![dog](https://github.com/KennCoder7/LearnToPayAttention/blob/main/dog.png)  

Due to the limited computational power, the VGG-type CNN net is condensed as follow:  
```
C64*3*3-C128*3*3-C256*3*3-P2*2-(Att1)-C512*3*3-P2*2-(Att2)-C512*3*3-P2*2-(Att3)-C512*3*3-P2*2-C512*3*3-P2*2-L512-L10
```
# Require
pytorch == 1.2.0

# Run
1. run vgg.py for pretraining the VGG net.  
2. run vgg_att.py for training the proposed net and visualizing the attention map.
