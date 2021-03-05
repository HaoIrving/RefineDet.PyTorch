# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/srn_1e3/


# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3/ --batch_size 16 
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_5l/ --batch_size 10
# CUDA_VISIBLE_DEVICES=3,2 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3/ --batch_size 16 --ngpu 2
# python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3/ --batch_size 16 
# CUDA_VISIBLE_DEVICES=3,2 python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3/ --batch_size 16 --ngpu 2

python train_refinedet.py --num_workers 12 --lr 1e-3 --save_folder weights/align_1e3_512res50/ --model 512_ResNet_50 --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/align_4e3_512res50/ --model 512_ResNet_50 --batch_size 16

# CUDA_VISIBLE_DEVICES=3,2 python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3_res101/ --batch_size 16 --ngpu 2
# python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3_512res101/ --model 512_ResNet_101 --batch_size 16

# python train_refinedet.py --num_workers 12 --lr 1e-3 --save_folder weights/align_1e3_1024res101/ --model 1024_ResNet_101 --batch_size 12
# python train_refinedet.py --num_workers 12 --lr 2e-3 --save_folder weights/align_2e3_1024rnx152/ --model 1024_ResNeXt_152 --batch_size 16
python eval_refinedet_coco.py
# git pull origin master

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srn_4e3/  #--ngpu 2 --batch_size 16
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srnv2_4e3/ # --ngpu 2 --batch_size 16

# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/srnv3_4e3/