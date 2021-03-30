
# git pull origin master
# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_4e3_05/ --ngpu 2 --batch_size 16
# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_1/ --ngpu 2 --batch_size 16 -aw 1
# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_5125vggbn/ --ngpu 2 --batch_size 16 --model 5125_vggbn
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_5125vggbn/ --ngpu 4 --batch_size 16 --model 5125_vggbn
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_1_ce_sigma1/ --batch_size 16 -aw 1 -atce -atsg 1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_1_ce_sigma02/ --batch_size 16 -aw 1 -atce -atsg 0.2
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01/ --ngpu 4 --batch_size 16 -aw 0.1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_4e3/ --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at_4e3/ --batch_size 16
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_4e3_01/ --batch_size 16 -aw 0.1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_4e3_1/ --ngpu 4 --batch_size 16 -aw 1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_4e3_01/ --ngpu 4 --batch_size 16 -aw 0.1
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_4e3_03/ --ngpu 4 --batch_size 16 -aw 0.3

# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_640vggbn_mo1/ --ngpu 4 --batch_size 16 --model 640_vggbn -mo
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn

# CUDA_VISIBLE_DEVICES=1 python eval_refinedet_coco.py --prefix weights/at1_mh_4e3_01_640vggbn_mo1 --model 640_vggbn -mo
# CUDA_VISIBLE_DEVICES=1 python eval_refinedet_coco.py --prefix weights/at1_mh_4e3_01_640vggbn --model 640_vggbn 
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at1_mh_4e3_01_5125vggbn --model 5125_vggbn 

# CUDA_VISIBLE_DEVICES=2,3 python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_5126vggbn/ --ngpu 2 --batch_size 16 --model 5126_vggbn
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at1_mh_4e3_01_5126vggbn --model 5126_vggbn 
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_mh_4e3_01_5126vggbn_mo/ --ngpu 4 --batch_size 16 --model 5126_vggbn -mo
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at1_mh_4e3_01_5126vggbn_mo --model 5126_vggbn -mo


### dcn head ###
# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_d_mh_4e3_1_ce_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn -atce -dcn
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at1_d_mh_4e3_1_ce_640vggbn --model 640_vggbn -dcn

# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_d_mh_sl_4e3_1_ce_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn -atce -sl
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at1_d_mh_sl_4e3_1_ce_640vggbn --model 640_vggbn -sl

python train_refinedet_postprocess.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_pp_wo_armconf_4e3_1_ce_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn -atce -at2
CUDA_VISIBLE_DEVICES=0 python eval_refinedet_coco_postprocess.py --prefix weights/at2_mh_pp_wo_armconf_4e3_1_ce_640vggbn --model 640_vggbn -at2

python train_refinedet_postprocess.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_mh_pp_wo_armconf_filter_4e3_1_ce_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn -atce -at2 -wof
CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco_postprocess.py --prefix weights/at2_mh_pp_wo_armconf_filter_4e3_1_ce_640vggbn --model 640_vggbn -at2


# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_d_mh_4e3_1_ce_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn -atce -dcn -at2
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at2_d_mh_4e3_1_ce_640vggbn --model 640_vggbn -dcn -at2

# python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at2_d_mh_sl_4e3_1_ce_640vggbn/ --ngpu 4 --batch_size 16 --model 640_vggbn -atce -sl -at2
# CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at2_d_mh_sl_4e3_1_ce_640vggbn --model 640_vggbn -sl -at2
