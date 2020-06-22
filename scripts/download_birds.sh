mkdir datasets
echo 'Downloading birds... about 4GB'
gdown https://drive.google.com/uc?id=1fCspnLiKAjOe7JKqAH8KDbcvt4FS__g6
tar -xzf birds.tar.gz -C datasets/
rm birds.tar.gz

echo 'Downloading inception model birds... 362MB'
gdown https://drive.google.com/uc?id=1Fn6JgALnESsIxPw34-s2saHtA1gc4bfF
tar -xzf inception_finetuned_models.tar.gz -C evaluation/inception_score
rm inception_finetuned_models.tar.gz