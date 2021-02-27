
# git pull origin master
# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_4e3_05/ --ngpu 2 --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_4e3/ --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at_4e3/ --batch_size 16
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_4e3_05/ --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/at_2e3/ --batch_size 16
python eval_refinedet_coco.py
