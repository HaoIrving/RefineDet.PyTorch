# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/srn_1e3/


# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/lr_1e3/ 
# python train_refinedet.py --num_workers 16 --lr 5e-4 --save_folder weights/lr_5e4/
# python train_refinedet.py --num_workers 16 --lr 3e-3 --save_folder weights/lr_3e3/
# python eval_refinedet_coco.py
# git pull origin master

# git fetch --tags

python train_sanet.py --num_workers 12 --lr 2e-3 --save_folder weights/solo_cs_fcos_2e3/ --batch_size 16
python eval_sanet_coco.py