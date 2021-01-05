# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/srn_1e3/


# python train_refinedet.py --num_workers 16 --lr 1e-3 --save_folder weights/lr_1e3/ 
# python train_refinedet.py --num_workers 16 --lr 5e-4 --save_folder weights/lr_5e4/
python train_refinedet.py --num_workers 16 --lr 3e-3 --save_folder weights/lr_3e3/
# python eval_refinedet_coco.py
git pull origin master

python train_refinedet.py --num_workers 16 --lr 3e-3 --save_folder weights/srn_3e3/
