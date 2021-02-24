# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/srn_1e3/


# python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3/ 
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3/ 
# python train_refinedet.py --num_workers 16 --lr 5e-4 --save_folder weights/lr_5e4/
# python train_refinedet.py --num_workers 16 --lr 3e-3 --save_folder weights/lr_3e3/
python eval_refinedet_coco.py
# git pull origin master

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srn_4e3/  #--ngpu 2 --batch_size 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srnv2_4e3/ # --ngpu 2 --batch_size 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srnv3_4e3/