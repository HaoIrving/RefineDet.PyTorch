
# SANet b16 lr 4e3 cps
python train_refinedet_ablation.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_sh_4e3_1_focal_512vggbn_woalign/ --ngpu 4 --batch_size 16 --model 512_vggbn -woalign
CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco_ablation.py --prefix weights/at1_sh_4e3_1_focal_512vggbn_woalign --model 512_vggbn -woalign


# SANet + deform conv
python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/at1_sh_4e3_1_focal_512vggbn/ --ngpu 4 --batch_size 16 --model 512_vggbn 
CUDA_VISIBLE_DEVICES=2 python eval_refinedet_coco.py --prefix weights/at1_sh_4e3_1_focal_512vggbn --model 512_vggbn 


