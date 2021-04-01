############ voc 07 ##########

python train_refinedet.py --num_workers 12 --lr 4e-3 --save_folder weights/voc_4e3_512vggbn/ --ngpu 4  --model 512_vggbn --batch_size 16 --dataset VOC -max 240
CUDA_VISIBLE_DEVICES=3 python eval_refinedet_voc07.py --prefix weights/voc_4e3_512vggbn  --model 512_vggbn 
