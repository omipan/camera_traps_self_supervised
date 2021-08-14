
# simclr variants
#

# --train loss can be simclr, triplet, simsiam,  rand_init,imagenet,supervised
# --pos_type augment_self, seq_positive, context_sample 
python main.py --train_loss simclr --not_cached_images --pos_type augment_self --backbone resnet18 --im_res 112 --batch_size 32 --epochs 2 --dataset cct20 --exp_name "simclr standard" 
python main.py --train_loss simclr --not_cached_images --pos_type seq_positive --backbone resnet18 --im_res 112 --batch_size 32 --epochs 2 --dataset cct20 --exp_name "simclr seq positive"  
python main.py --train_loss simclr --not_cached_images --pos_type context_sample --backbone resnet18 --im_res 112 --batch_size 32 --epochs 2 --dataset cct20 --exp_name "simclr context distance"   
