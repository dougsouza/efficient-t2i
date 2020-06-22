mkdir logs
echo 'Downloading pretrained model for birds... 776MB'
gdown https://drive.google.com/uc?id=1YLqAkHuyPWof64amelOie2t2NwRdmsWk
tar -xzf EfficientGAN_96\[ICJNN.exp1\]\[Interp.ttur\]_CUB_0114_1725.tar.gz -C logs/
rm EfficientGAN_96\[ICJNN.exp1\]\[Interp.ttur\]_CUB_0114_1725.tar.gz