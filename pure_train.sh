ulimit -n 32768
CUDA_VISIBLE_DEVICES=2 python train2.py --eval -s data/VastGaussian/Residence/residence-pixsfm/train -m output/residence/random --port 6888
# CUDA_VISIBLE_DEVICES=3 python train.py --eval -s data/VastGaussian/Rubble/rubble-pixsfm/train -m output/rubble/debug3 --port 6888