# ijcnn exps
name='EfficientGAN_96[ICJNN.exp1][Interp.ttur]'
ngpu=1
dataset='Oxford'

python -m torch.distributed.launch \
    --master_port=6997 --nproc_per_node=${ngpu}  \
    train.py ${name} \
    --dataset=${dataset} \
    --batch_size=8 \
    --d_model=biggan_deep \
    --ndf=96 \
    --d_lr=0.0004 \
    --d_steps=1 \
    --g_model=biggan_deep \
    --ngf=96 \
    --g_lr=0.0001 \
    --EMA_G \
    --weight_init=orthogonal \
    --n_epochs=600 \
    --save_checkpoint_ep=20 \
    --ngpu=${ngpu} \
    --num_workers=4 \
    --n_samples=16 \
    --log_step=10 \
    --save_sample_step=200 \
    --prior=normal \
    --z_dim=128 