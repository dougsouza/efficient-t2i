OUT_DIR='datasets/CUB'
mkdir -p $OUT_DIR
echo 'Downloading birds... about 4GB'
# download images
gdown https://drive.google.com/uc?id=13w8kckfJTWb4LD0Xf4J0huvgWmFLKpb9
tar -xzf images.tar.gz -C $OUT_DIR
rm images.tar.gz
# download attn embeddings
gdown https://drive.google.com/uc?id=1Si5SRU7883wgCU3NZ-QDwoj3YoALUlan
tar -xzf attn_embeddings.tar.gz -C $OUT_DIR
rm attn_embeddings.tar.gz
# download captions
gdown https://drive.google.com/uc?id=1hZXdCetIVsWR_LyL_rcQzP_NLmSu_Xcs --output $OUT_DIR/train/
gdown https://drive.google.com/uc?id=1xyh3EKe372xJgJXkShN73fHgShspM0Ca --output $OUT_DIR/test/

# echo 'Downloading inception model birds... 362MB'
gdown https://drive.google.com/uc?id=1Fn6JgALnESsIxPw34-s2saHtA1gc4bfF
tar -xzf inception_finetuned_models.tar.gz -C evaluation/inception_score
rm inception_finetuned_models.tar.gz
