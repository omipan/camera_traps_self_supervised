import numpy as np
import os
import json
import argparse

import torch
import torch.nn.functional as F
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader, Subset

import utils as ut
import datasets as ds
from models import EmbModel


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Self-Supervised with Context')
    parser.add_argument('--dataset', default='cct20', choices=['cct20', 'kenya', 'icct','serengeti','fmow'], type=str)
    parser.add_argument('--model_file', type=str)
    
    ip_args = vars(parser.parse_args())
    
    
    # Loading saved model
    checkpoint = torch.load(ip_args['model_file'])
    args = checkpoint['args']

    print('\n**********************************')
    print('Experiment   :', args['exp_name'])
    print('Results file :', args['op_res_name'])
    print('Model file   :', ip_args['model_file'])
    print('Dataset      :', ip_args['dataset'])
        

    args['dataset'] = ip_args['dataset']
    args['data_dir'] = os.path.join('cam_data/', args['dataset'], '')
    args['metadata'] = os.path.join(args['data_dir'], args['dataset']+'_context_final.csv')
    args['cache_images'] = False
    args['device'] = 'cuda'
    
    # fix as older saved models might not have this  
    args['return_oracle_pos_same_loc'] = False
    
    
    # load weights into standard resnet
    # base_encoder = eval(args['backbone'])
    # model = base_encoder(pretrained=False)
    # model.fc = torch.nn.Identity()
    # 
    # state_dict = {k.replace("enc.", ""): v for k, v in checkpoint['state_dict'].items()}
    # state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    # msg = model.load_state_dict(state_dict, strict=True)

    # load weights into our model
    model = EmbModel(eval(args['backbone']), args).to(args['device'])
    msg = model.load_state_dict(checkpoint['state_dict'], strict=True)


    # get dataset
    train_set, train_set_lin, test_set_lin, train_inds_lin_1, train_inds_lin_10 = ds.get_dataset(args)
    args['context_size'] = train_set.context_size
    args['num_train'] = train_set.num_examples
        

    # data loaders - used for linear evaluation 
    train_loader_lin = DataLoader(train_set_lin, batch_size=args['batch_size'], num_workers=args['workers'], shuffle=False)
    test_loader_lin  = DataLoader(test_set_lin,  batch_size=args['batch_size'], num_workers=args['workers'], shuffle=False)        


    print('\nLinear evaluation - class')
    train_inds = [np.array(train_inds_lin_1), np.array(train_inds_lin_10), np.arange(len(train_set_lin))]
    train_split_perc = [1, 10, 100]
    res = ut.linear_eval_all(model, train_loader_lin, test_loader_lin, args, train_inds, train_split_perc, True, 'target')


    print('\nLinear evaluation - location id')
    train_inds = [np.array(train_inds_lin_1), np.array(train_inds_lin_10), np.arange(len(train_set_lin))]
    train_split_perc = [1, 10, 100]
    res = ut.linear_eval_all(model, train_loader_lin, test_loader_lin, args, train_inds, train_split_perc, True, 'location_id')
