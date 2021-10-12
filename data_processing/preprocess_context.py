import numpy as np
import pickle
import json
import pandas as pd
import random
import datetime
import argparse

original_data_dir = '/media/GRUMPY_HDD/omi_data/cct20/'
## image folders under /cct20/ should be in IMAGE LOADER  

def main(args):

    ## Get cropped image path information (assumes preprocess_images.py has already run)

    # data split strategy may vary per dataset
    if args['dataset'] in ['cct20','icct']:
        img_folders = ['train_images','cis_val_images','trans_val_images','cis_test_images','trans_test_images']
    elif args['dataset'] in ['serengeti','fmow']:
        img_folders = ['train','test']
    else:
        img_folders = ['train','val','test']

    path_to_box = pd.DataFrame(columns=['img_name','x1','x2','y1','y2','width','height','img_path','img_set'])
    for folder in img_folders:
        path = args['image_dir']+folder+'/{}_path_to_box.csv'.format(args['dataset'])
        folder_df = pd.read_csv(path,index_col=0)
        folder_df['img_path'] = folder+'/'+folder_df['img_name']
        folder_df['img_set'] = folder
        path_to_box = path_to_box.append(folder_df)
    path_to_box['img_id'] = path_to_box.img_name.apply(lambda x: x.split('_')[0])
    for val in ['x1','x2','y1','y2','width','height']:
        path_to_box[val] = path_to_box[val].astype(int)
    path_to_box['area'] = path_to_box['width']*path_to_box['height']

    ## Load annotation information (not to be used during self-supervised stage)
    with open('{}{}'.format(args['original_data_dir'],args['annotation_file'])) as f:
        info_cct = json.load(f)
    cct_ann_df = pd.DataFrame(info_cct['annotations'])

    cct_ann_df['bbox'] = cct_ann_df['bbox'].apply(lambda x:x if type(x)==list else [0.0,0.0,0.0,0.0])
    cct_ann_df['x1'] = cct_ann_df['bbox'].apply(lambda x: int(x[0]))
    cct_ann_df['y1'] = cct_ann_df['bbox'].apply(lambda x: int(x[1]))
    cct_ann_df['width'] = cct_ann_df['bbox'].apply(lambda x: int(x[2]))
    cct_ann_df['height'] = cct_ann_df['bbox'].apply(lambda x: int(x[3]))

    ## merge annotation info with box path info
    cct_ann_df = cct_ann_df.merge(path_to_box,left_on=['image_id','x1','y1','width','height'],right_on=['img_id','x1','y1','width','height'],how='left')

    ## Load image-level information
    with open('{}{}'.format(args['original_data_dir'],args['annotation_file'])) as f:
        info_cct = json.load(f)
    cct_info_df = pd.DataFrame(info_cct['images'])
    cct_info_df = cct_info_df.rename(columns={'height':'img_height','width':'img_width'})
    cct_info_df.set_index('id',inplace=True)

    cct_info_df['datetime'] = pd.to_datetime(cct_info_df.date_captured)
    cct_info_df['time'] = pd.to_datetime(cct_info_df.date_captured).dt.time
    
    ## merge annotation info with image-level info
    context_info = cct_ann_df.merge(cct_info_df,left_on='image_id',right_index=True,how='left')
    context_info.drop(columns=['image_id','date_captured'],inplace=True)
    ## drop images with no boxes
    context_info.dropna(axis=0,subset=['img_name'],inplace=True)

    context_info['datetime'] = pd.to_datetime(context_info['datetime'])
    context_info['species'] = context_info.img_name.apply(lambda x: '_'.join(x.split('_')[2:]).replace('.jpg',''))
    
    ## define same sequence images (previous or next burst taken a short period of time around a given image) to use as positives
    sequence_df = context_info.copy()
    sequence_df = sequence_df[['img_id','location','datetime','img_name','img_path']].drop_duplicates()
    sequence_df.sort_values(['location','datetime'],inplace=True)
    
    seconds_gap = datetime.timedelta(seconds=5) ## max distance (in seconds) for images to be considered bursts of the same sequence
    clone_df = sequence_df.copy()

    count=0
    for idx,row in sequence_df.iterrows():
        if count%1000==0:
            print(count/len(sequence_df))
        count+=1
        seq_start = row.datetime-seconds_gap
        seq_end = row.datetime+seconds_gap
        seq_location = row.location
        clone_df.loc[(clone_df.location==seq_location) & (clone_df.datetime.between(seq_start,seq_end)) & (clone_df.img_name!=row.img_name),'time_diff'] = np.abs(clone_df.loc[(clone_df.location==seq_location) & (clone_df.datetime.between(seq_start,seq_end)) & (clone_df.img_name!=row.img_name)].datetime - row.datetime)
        
        img_before = clone_df.loc[(clone_df.location==seq_location) & (clone_df.datetime.between(seq_start,seq_end)) & (clone_df.img_id!=row.img_id) &(clone_df.datetime < row.datetime)].sort_values('time_diff')
        img_before = [] if img_before.empty else [img_before.groupby('img_id').time_diff.min().sort_values().index[0]]
        img_after = clone_df.loc[(clone_df.location==seq_location) & (clone_df.datetime.between(seq_start,seq_end)) & (clone_df.img_id!=row.img_id) &(clone_df.datetime > row.datetime)].sort_values('time_diff')
        img_after = [] if img_after.empty else [img_after.groupby('img_id').time_diff.min().sort_values().index[0]]
        seq_img_ids = img_after + img_before
        same_img_other_crops = list(clone_df.loc[(clone_df.img_id==row.img_id) &(clone_df.img_name!=row.img_name)].img_path)
        same_seq_img_paths = list(set(list(clone_df.loc[clone_df.img_id.isin(seq_img_ids)].img_path)+same_img_other_crops))
        if len(same_seq_img_paths)>0:
            clone_df.at[idx, 'prev_same_next_img_boxes'] = same_seq_img_paths
    clone_df['prev_same_next_img_boxes'] = clone_df['prev_same_next_img_boxes'].fillna({i: [] for i in clone_df.index})

    ## add sequence information in metadata
    context_info = context_info.merge(clone_df[['img_name','prev_same_next_img_boxes']],left_on='img_name',right_on='img_name',how='left')
    img_boxes = context_info.groupby(['img_id']).img_path.nunique().reset_index().rename(columns={'img_path':'boxes_per_img_id'})
    context_info = context_info.merge(img_boxes,left_on='img_id',right_on='img_id',how='left')
    context_info['prev_same_next_img_boxes']=context_info.prev_same_next_img_boxes.apply(lambda x: x if type(x)==list else list(x) )
    context_info['boxes_per_prev_same_next_img'] = context_info.prev_same_next_img_boxes.apply(lambda x: len(x))

    ## sample 1,10% subset of training data (for semi-supervised stage)
    context_info['is_in_train_1perc'] = False
    context_info['is_in_train_10perc'] = False
    species_list = list(context_info.query('img_set=="train"').species.unique())
    for perc in [0.01,0.1]:
        context_info['is_in_train_{}perc'] = False
        for species in species_list:
            sub_df = context_info.query('img_set=="train"').query('species==@species')
            random.seed(args['seed']) # change to get a different subsample per species
            species_filenames = random.sample(set(sub_df.file_name), int(np.ceil(len(sub_df)*perc)))
            context_info.loc[context_info.file_name.isin(species_filenames),'is_in_train_{}perc'.format(int(100*perc))] = True
    
    ### save context file to be used during training
    context_info.to_csv('{}{}_context_file.csv'.format(args['image_dir'],args['dataset']))

'''python preprocess_context.py --dataset cct20 --annotation_file CaltechCameraTrapsECCV18.json'''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Metadata dataframe generation for SSL pretraining')
    
    parser.add_argument('--dataset', default='cct20', choices=['cct20', 'kenya', 'icct','serengeti','fmow'], type=str)
    parser.add_argument('--annotation_file', default='CaltechCameraTrapsECCV18.json', type=str)
    parser.add_argument('--seed', default=42, type=int)
    #parser.add_argument('--area_thr', default=4096, type=float) # 64x64=4096
    
    # turn the args into a dictionary
    args = vars(parser.parse_args())

    args['original_data_dir'] = '/media/GRUMPY_HDD/omi_data/{}/'.format(args['dataset']) ## should point to where you have downloaded the data
    args['image_dir'] = 'cam_data/{}/'.format(args['dataset'])
    
    main(args)


    