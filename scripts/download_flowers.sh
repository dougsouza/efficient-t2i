OUT_DIR='datasets/Oxford102'
mkdir -p $OUT_DIR
echo 'Downloading flowers... about 3GB'
# download images
gdown https://drive.google.com/uc?id=1RiHiawgWhk6DbirOgIX7QH27fq0cJyrK
tar -xzf images.tar.gz -C $OUT_DIR
rm images.tar.gz
# download captions
gdown https://drive.google.com/uc?id=1GvYTVlfZku0ym6o7_EFrpMmpZwMTN-ky --output $OUT_DIR/train/
gdown https://drive.google.com/uc?id=1HVpmumbCnRD1ifTU2H-JKfF8v1RhnZ2f --output $OUT_DIR/test/

