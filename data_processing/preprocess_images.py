import argparse
import time
import os
import json
import copy
import random
import numbers
from tqdm import tqdm
from pycocotools.coco import COCO
import cv2
import pandas as pd
from PIL import Image
import numpy as np

from utils import modify_polygon_coords,image_load_cv2

class Annotation_Filter():

    def __init__(self,
                 coco_dataset):
        self.coco_obj = coco_dataset
        self.imgIds = coco_dataset.getImgIds()
        self.annIds = coco_dataset.getAnnIds()
        print('\nTotal images: {}'.format(len(self.imgIds)))
        print('Total annotations: {}'.format(len(self.annIds)))

    

    def keep_ann_necessary(self):
        img_annIds = self.coco_obj.getAnnIds(imgIds=self.imgIds)
        self.annIds = list(set(self.annIds).intersection(set(img_annIds)))

    def keep_img_necessary(self):

        filtered_images = []
        for im_id in self.imgIds:
            keep_image = False
            for ann_object in self.coco_obj.imgToAnns[im_id]:
                if ann_object['id'] in self.annIds:
                    keep_image = True
            if keep_image:
                filtered_images.append(im_id)

        self.imgIds = filtered_images



    def filter_out_small_area(self,
                              area_thr=0):
        """Function that filters the imgIds and the annIds based on the area of the bounding box
        Args:
            area_thr: area threshold for the box to filter out the small/blurry/distant annotations
        Returns:
            Filtered set of images and annotations based on the box area
        """
        print('\nFILTER SMALL ANNOTATIONS')
        filtered_images = []
        filtered_anns = []
        for im_id in self.imgIds:
            keep_image = False
            for ann_object in self.coco_obj.imgToAnns[im_id]:
                if 'area' not in ann_object:
                    ann_object['area'] = ann_object['bbox'][2]*ann_object['bbox'][3]

                if ann_object['area'] >= area_thr and ann_object['id'] in self.annIds:
                    keep_image = True
                    # filteredAnns[ann['id']] = ann
                    filtered_anns.append(ann_object['id'])
            if keep_image:
                filtered_images.append(im_id)
                # filteredImages[im_id] = self.coco_obj.imgs[im_id]

        self.imgIds = filtered_images
        self.annIds = filtered_anns
        print("\nNumber of images containing boxes with area (>{}) : {}".format(area_thr, len(self.imgIds)))
        print("Number of annotations in boxes with area (>{}) : {}".format(area_thr, len(self.annIds)))
        self.keep_ann_necessary()
        print("Number of images containing boxes with area (>{}) : {}".format(area_thr, len(self.imgIds)))
        print("Number of annotations in boxes with area (>{}) : {}".format(area_thr, len(self.annIds)))

    def keep_anns_with_boxes(self):
        """Function that filters the imgIds and the annIds based on the availability of bounding box
                Returns:
                    Filtered set of images and annotations based on the existence of box around object
                """
        print('\nFILTER OUT ANNOTATIONS WITH NO BOXES')

        ann_ids_with_box = []
        for ann_id in self.annIds:
            ann_object = self.coco_obj.anns[ann_id]
            if 'bbox' in ann_object:
                ann_ids_with_box.append(ann_id)

        print('\nTotal annotations: {} | Reduced : {}'.format(len(self.annIds), len(ann_ids_with_box)))

        filtered_images = []
        filtered_anns = []
        for im_id in self.imgIds:
            keep_image = False
            for ann_object in self.coco_obj.imgToAnns[im_id]:
                if ann_object['id'] in ann_ids_with_box:
                    keep_image = True
                    # filteredAnns[ann['id']] = ann
                    filtered_anns.append(ann_object['id'])
            if keep_image:
                filtered_images.append(im_id)
                # filteredImages[im_id] = self.coco_obj.imgs[im_id]

        self.imgIds = filtered_images
        self.annIds = filtered_anns
        print("\nNumber of images containing boxes: {}".format(len(self.imgIds)))
        print("Number of annotations containing boxes: {}".format(len(self.annIds)))

        self.keep_ann_necessary()
        print("\nNumber of images containing boxes: {}".format(len(self.imgIds)))
        print("Number of annotations containing boxes: {}".format(len(self.annIds)))



    

class BoxCropper(object):

    def __init__(self,
                 coco_dataset,
                 dataset_style,
                 annIds_to_keep,
                 root_dir,
                 save_crops=False,
                 output_root_dir=None
                 ):

        """
         Data transform class that crops the bounding box of interest out of the image
        Args:
        sample: sample image with the corresponding label
        bbox: bounding box coordinates in the order x,y,w,h where
        x is distance from left, y is distance from top, w is bounding boxs width and h is
        bounding boxs height. The values of the bounding box corrdinates range in the images
        dimensions (not normalized to ratio).
        Returns: Cropped image based on the defined bounding box coordinates
         """
        self.coco_obj = coco_dataset
        self.annIds_to_keep = annIds_to_keep
        self.root_dir = root_dir
        self.save_crops = save_crops
        self.output_root_dir = output_root_dir


        self.dataset = dataset_style

    def crop_image(self, img_id_list):

        path_to_box = []
        for im_id in tqdm(img_id_list):
            img_annIds = self.coco_obj.getAnnIds(imgIds=[im_id])
            img_annIds = list(set(img_annIds).intersection(set(self.annIds_to_keep)))

            coco_image_object = self.coco_obj.imgs[im_id]

            # define a converter from category id to species
            category_id_to_species = {}
            for an in self.coco_obj.cats.values():
                category_id_to_species[an['id']] = an['name']

            # find BH image server directory
            img_file = coco_image_object['file_name']
            input_path = self.root_dir + img_file

            # load biome health image
            current_image = image_load_cv2(input_path)
            if current_image is not None:
                if self.save_crops:
                    output_dir = self.output_root_dir
                    os.makedirs(output_dir, exist_ok=True)

                    # img characteristics
                    img_height = coco_image_object['height']
                    img_width = coco_image_object['width']

                    # crop all image annotations

                    for num_box, img_ann_id in enumerate(img_annIds):
                        ann_object = self.coco_obj.anns[img_ann_id]

                        x1, y1, width, height = ann_object['bbox']

                        if (isinstance(x1, numbers.Number) and isinstance(y1, numbers.Number)
                                and isinstance(width, numbers.Number) and isinstance(height, numbers.Number)):
                            x1 = int(x1)
                            y1 = int(y1)
                            width = int(width)
                            height = int(height)
                            x2 = x1 + width
                            y2 = y1 + height

                        crop_img = current_image[y1:y2, x1:x2].copy()
                        species = category_id_to_species[ann_object['category_id']]
                       

                        new_row = {'img_name':'{}_{}_{}.jpg'.format(str(im_id),num_box,species),
                                             'x1':x1,'x2':x2,'y1':y1,'y2':y2,'width':width,'height':height}

                        path_to_box.append(new_row)
                        if self.save_crops:
                            output_path = '{}_{}_{}.jpg'.format(output_dir + str(im_id),
                                                                num_box,
                                                                species)
                            cv2.imwrite(output_path, crop_img)

        pd.DataFrame(path_to_box).to_csv(self.output_root_dir+'{}_path_to_box.csv'.format(self.dataset))
        print('Boxes cropped out of Images')

def main(args):


    if args['dataset'] == 'cct20':
       files_to_process = os.listdir(args['input_root_dir']+'eccv_18_annotation_files/')
       for file in files_to_process:
        folder_name = file.replace('annotations.json','images/')
        # Dataset load and modification
        filepath = args['input_root_dir']+'eccv_18_annotation_files/'+file
        dataset = COCO(filepath)
        # FILTER ANNOTATIONS AND IMAGES
        annFilter = Annotation_Filter(dataset)
        annFilter.keep_anns_with_boxes()  # keep annotations with boxes
        annFilter.filter_out_small_area(area_thr=args['area_thr'])  # threshold for min area

        final_annIds = annFilter.annIds
        final_imgIds = annFilter.imgIds

        print('Final filtered number of annotations: {} \n and images: {}'.format(len(final_annIds), len(final_imgIds)))
        print(args['output_root_dir']+folder_name)
        # CROP BOXES OUT OF ANNOTATED IMAGES
        BoxCrop = BoxCropper(dataset,
                             dataset_style=args['dataset'],
                             annIds_to_keep=final_annIds,
                             root_dir=args['input_root_dir']+'eccv_18_cropped/',
                             save_crops=True,  # save crops to following dir
                             output_root_dir=args['output_root_dir']+folder_name)

        BoxCrop.crop_image(img_id_list=final_imgIds)


    elif args['dataset'] == 'kenya':


        # Dataset load and modification
        dataset = COCO(args['ann_coco_file'])
        dataset_mod = modify_polygon_coords(dataset)

        # FILTER ANNOTATIONS AND IMAGES
        annFilter = Annotation_Filter(dataset_mod)

        # Get rid of empty and person, vehicle annotations (i.e. keep only animal boxes)

        annFilter.filter_out_small_area(area_thr=args['area_thr']) # threshold for min area
    
        final_annIds = annFilter.annIds
        final_imgIds = annFilter.imgIds


        print('Final filtered number of annotations: {} \n and images: {}'.format(len(final_annIds), len(final_imgIds)))

        # CROP BOXES OUT OF ANNOTATED IMAGES
        BoxCrop = BoxCropper(dataset_mod,
                             dataset_style=args['dataset'],
                             annIds_to_keep=final_annIds,
                             root_dir=args['input_root_dir'],
                             save_crops=True, # save crops to following dir
                             output_root_dir=args['output_root_dir']
                             )
        BoxCrop.crop_image(img_id_list=final_imgIds)

'''python preprocess_images.py --dataset cct20  '''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Image filtering and cropping before SSL training')
    
   
    parser.add_argument('--dataset', default='cct20', choices=['cct20', 'kenya', 'icct','serengeti','fmow'], type=str)
    parser.add_argument('--area_thr', default=4096, type=float) # 64x64=4096
    parser.add_argument('--train_from_megadetector', action='store_true')
    parser.add_argument('--exp_name', default='', type=str)
    
    
    # turn the args into a dictionary
    args = vars(parser.parse_args())


    args['data_dir'] = '/media/GRUMPY_HDD/omi_data/'
    args['input_root_dir'] = args['data_dir'] + args['dataset'] +'/'
    args['output_root_dir'] = 'cam_data/{}/'.format(args['dataset'])
    args['metadata'] = os.path.join(args['output_root_dir'], args['dataset']+'_context_file.csv')
    if args['dataset'] == 'kenya':
         args['ann_coco_file'] = args['input_root_dir']+'kenya_coco.json'

    # objects_of_interest = ['animal'] # list of coco objects to keep

    main(args)