ulimit -n 32768
# CUDA_VISIBLE_DEVICES=3 python train2.py --eval -s data/VastGaussian/Residence/residence-pixsfm/train -m output/residence/trying_debug --port 7888
# CUDA_VISIBLE_DEVICES=3 python train2.py --eval -s data/VastGaussian/Rubble/rubble-pixsfm/train -m output/rubble/trying_debug --port 7888

# CUDA_VISIBLE_DEVICES=2 python train2.py --eval -s data/tandt_db/tandt/truck -m output/truck/adding_allpnts_pt_meanfeatlearn_1e3 --port 7888
# CUDA_VISIBLE_DEVICES=3 python train2.py --eval -s data/tandt_db/tandt/truck -m output/truck/adding_allpnts_oct_meanfeatlearn_1e3 --port 6888

CUDA_VISIBLE_DEVICES=2 python train2.py --eval -s data/tandt_db/tandt/truck -m output/truck/adding_allpnts_pt_allparams --port 2888
# CUDA_VISIBLE_DEVICES=3 python train2.py --eval -s data/tandt_db/tandt/truck -m output/truck/adding_vis_pt_allparams --port 1888