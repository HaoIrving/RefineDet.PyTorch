
# git pull origin master
# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_4e3_05/ --ngpu 2 --batch_size 16
# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_1/ --ngpu 2 --batch_size 16 -aw 1
CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_5125vggbn/ --ngpu 2 --batch_size 16 --model 5125_vggbn
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_1_ce_sigma1/ --batch_size 16 -aw 1 -atce -atsg 1
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_1_ce_sigma02/ --batch_size 16 -aw 1 -atce -atsg 0.2
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01/ --ngpu 4 --batch_size 16 -aw 0.1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_4e3/ --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at_4e3/ --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_4e3_01/ --batch_size 16 -aw 0.1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_4e3_1/ --ngpu 4 --batch_size 16 -aw 1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_4e3_01/ --ngpu 4 --batch_size 16 -aw 0.1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_4e3_03/ --ngpu 4 --batch_size 16 -aw 0.3
CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py
