ulimit -n 32768
# CUDA_VISIBLE_DEVICES=3 python train3.py --eval -s data/VastGaussian/Residence/residence-pixsfm/train -m output/residence/trying_debug --port 7888
# CUDA_VISIBLE_DEVICES=3 python train3.py --eval -s data/VastGaussian/Rubble/rubble-pixsfm/train -m output/rubble/trying_debug --port 7888
CUDA_VISIBLE_DEVICES=3 python train3.py --eval -s data/tandt_db/tandt/truck -m output/truck/trying_debug --port 7888