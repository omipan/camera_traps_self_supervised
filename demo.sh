

## STEP1: extract camera trap object regions from images (these can be either available from the given data or acquired from Megadetector (ref))
#python preprocess_images.py --dataset cct20

## STEP2: save a metadata file for the contextual information
#python preprocess_context.py --dataset cct20 --annotation_file CaltechCameraTrapsECCV18.json


## STEP 3: self-supervised training and evaluation
# simclr variants
#

# --train loss can be simclr, triplet, simsiam,  rand_init,imagenet,supervised
# --pos_type augment_self, seq_positive, context_sample 
python main.py --train_loss simclr --not_cached_images --pos_type augment_self --backbone resnet18 --im_res 112 --dataset cct20 --exp_name "simclr standard" 
python main.py --train_loss simclr --not_cached_images --pos_type seq_positive --backbone resnet18 --im_res 112 --dataset cct20 --exp_name "simclr seq positive"  
python main.py --train_loss simclr --not_cached_images --pos_type context_sample --backbone resnet18 --im_res 112 --dataset cct20 --exp_name "simclr context distance"   


