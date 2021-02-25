# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/srn_1e3/


# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3/ 
# CUDA_VISIBLE_DEVICES=3,2 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3/ --batch_size 16 --ngpu 2
python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3/ 
# CUDA_VISIBLE_DEVICES=3,2 python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3/ --batch_size 16 --ngpu 2
python eval_refinedet_coco.py
# git pull origin master

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srn_4e3/  #--ngpu 2 --batch_size 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srnv2_4e3/ # --ngpu 2 --batch_size 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srnv3_4e3/