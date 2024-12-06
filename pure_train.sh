ulimit -n 32768
CUDA_VISIBLE_DEVICES=3 python train.py --eval -s /data/zhangao/tandt_db/tandt/truck -m output/truck/sfm --port 7888