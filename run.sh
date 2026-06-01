CUDA_VISIBLE_DEVICES=$1 python hyrag.py --target_data cora --ckpt pretrained_graphclip --seed $2 --batch_size 32 --alpha $3 --beta $4 --hyper_k 3
