import logging
import random
import argparse
import numpy as np
import os
import json 
import datetime
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision.models import resnet18, resnet50

from models import EmbModel
import utils as ut
import datasets as ds
from losses import *
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    
        
def train(model, args, train_loader, optimizer, scheduler, epoch):

    model.train()
    loss_meter = ut.AverageMeter(args['train_loss'])
    train_bar = tqdm(train_loader)
    for data in train_bar:

        optimizer.zero_grad()
        x = torch.cat((data['im_t1'], data['im_t2']), 0).to(args['device'])
        b_size = x.shape[0]
        op = model(x)
        


        
        if args['return_context']:
            data['con'] = data['con'].to(args['device'])
            
        if args['store_embeddings']:
            model.update_memory(data['id'].to(args['device']), op['emb'][:b_size//2, :])
        
        if args['train_loss'] == 'supervised': 
            targets = torch.cat((data['target'], data['target']), 0).long().to(args['device'])
            loss = F.cross_entropy(op['emb'], targets)
        
        elif args['train_loss'] == 'simsiam':    
            p1 = op['emb_p'][:b_size//2, :]
            p2 = op['emb_p'][b_size//2:, :]
            z1 = op['emb'][:b_size//2, :]
            z2 = op['emb'][b_size//2:, :]
            loss = simsiam(p1, z1, p2, z2, args)
            
        elif args['train_loss'] == 'triplet':
            loss = triplet_loss(op['emb'], args, margin=args['triplet_margin'])

        elif args['train_loss'] == 'triplet_hard':
            loss = triplet_hard_loss(op['emb'], args, margin=args['triplet_margin'])
                
      
    
        elif args['train_loss'] == 'simclr':            
            loss = nt_xent(op['emb'][:b_size//2, :], op['emb'][b_size//2:, :], args)
            
  
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_meter.update(loss.item(), x.shape[0])
        train_bar.set_description("Train epoch {}, loss: {:.4f}".format(epoch, loss_meter.avg))

    return loss_meter.avg

        
def select_train_items(model, args, train_loader):
    # samples new positive items based on distance in context and embedding space
    
    if args['pos_type'] in ['image_emb_sample', 'context_and_image_emb_sample']:
        # distance in embedding space    
        emb = model.emb_memory
        emb_dist = emb@emb.t()  # already normalized
        emb_dist = torch.softmax(emb_dist / args['emb_temp_select'], dim=1)
    
    if args['pos_type'] in ['context_sample', 'context_and_image_emb_sample']:
        # distance in context space
        context = train_loader.dataset.context.to(args['device'])
        con_dist = torch.cdist(context, context) 
        con_dist = torch.softmax(-con_dist / args['con_temp_select'], dim=1)
       
    if args['pos_type'] == 'image_emb_sample': 
        dist = emb_dist
    elif args['pos_type'] == 'context_sample': 
        dist = con_dist
    elif args['pos_type'] == 'context_and_image_emb_sample':     
        dist = emb_dist*con_dist
        
    # sample new positives based on distance matrix
    sample_inds = torch.multinomial(dist, 1)[:, 0]

    train_loader.dataset.update_alternative_positives(sample_inds)

    # check how often the same class is picked - just for debugging
    targets = train_loader.dataset.targets
    acc = torch.tensor(targets) == torch.tensor(targets)[sample_inds.cpu()]
    acc = acc.float().mean()
    same_inds = (sample_inds.cpu() == torch.arange(sample_inds.shape[0])).float().mean()
    print(round(acc.item(), 3), round(same_inds.item(), 3))
    
    
def main(args):
    assert torch.cuda.is_available()
    cudnn.benchmark = True

    # get datasets
    train_set, train_set_lin, test_set_lin, train_inds_lin_1, train_inds_lin_10 = ds.get_dataset(args)
    args['context_size'] = train_set.context_size
    args['num_train'] = train_set.num_examples
    print('Running on: ',torch.cuda.get_device_name(torch.cuda.current_device()))  
    
    # for supervised 1% and 10% it will be a subset of the data
    if args['train_loss'] == 'supervised':
        args['projection_dim'] = train_set.num_classes
        
        if args['supervised_amt'] == 1:
            train_set = Subset(train_set, train_inds_lin_1)
        elif args['supervised_amt'] == 10:
            train_set = Subset(train_set, train_inds_lin_10)

    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True,
                              num_workers=args['workers'], drop_last=False)
            
    # data loaders - used for linear evaluation 
    train_loader_lin_1   = DataLoader(Subset(train_set_lin, train_inds_lin_1),
                                      batch_size=args['batch_size'], num_workers=args['workers'], shuffle=False)
    train_loader_lin_10  = DataLoader(Subset(train_set_lin, train_inds_lin_10), 
                                      batch_size=args['batch_size'], num_workers=args['workers'], shuffle=False)
    train_loader_lin_100 = DataLoader(train_set_lin, batch_size=args['batch_size'], 
                                      num_workers=args['workers'], shuffle=False)
    test_loader_lin      = DataLoader(test_set_lin,  batch_size=args['batch_size'], 
                                      num_workers=args['workers'], shuffle=False)        

    if args['pretrained_model'] != '':
        args['pretext_finetune'] = True
    
    # initialize model
    base_encoder = eval(args['backbone'])
    model = EmbModel(base_encoder, args).to(args['device'])
    
    if args['pretrained_model'] != '':
        # need to exlude projector as it will be a different size for supervised
        print('Loading pretrained', args['pretrained_model'])
        state_dict = torch.load(args['pretrained_model'])['state_dict']
        state_dict = {k: v for k, v in state_dict.items() if 'projector' not in k}
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg, '\n')
    
            
    # if burn in period, freeze the backbone weights for the first few epochs
    if args['burn_in'] > 0:
        for param in model.enc.parameters():
            param.requires_grad = False

    optimizer = torch.optim.SGD(
        model.parameters(),
        args['learning_rate'],
        momentum=args['momentum'],
        weight_decay=args['weight_decay'])

    # lr decay schedule
    if args['train_loss'] not in ['imagenet', 'rand_init']: 
        if args['schedule'] == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, args['epochs'] * len(train_loader))
                
        elif args['schedule'] == 'lambda':
            scheduler = LambdaLR(
                 optimizer,
                 lr_lambda=lambda step: get_lr(
                     step,
                     args['epochs'] * len(train_loader),
                     args['learning_rate'],
                     1e-3))
    

    # main train loop
    res = [] 
    for epoch in range(1, args['epochs'] + 1):  
        if args['burn_in'] == epoch:
            for param in model.enc.parameters():
                param.requires_grad = True

        if args['pos_type'] in ['context_sample', 'image_emb_sample', 'context_and_image_emb_sample']:
            # choose positives
            if epoch > args['burn_in_select']:
                select_train_items(model, args, train_loader)
    
        loss_avg = train(model, args, train_loader, optimizer, scheduler, epoch)  
        
        if epoch >= args['eval_interval'] and epoch % args['eval_interval'] == 0:
            test_acc, test_acc_bal = ut.linear_eval(model, train_loader_lin_1, test_loader_lin, args, ' 1%', False)
            res.append([epoch, test_acc, test_acc_bal])
            ut.plot_progress(res, args)
    
            # save checkpoint 
            op = {'state_dict':model.state_dict(), 'args':args, 'epoch':epoch}
            torch.save(op, args['op_file_name'])
    
    print('\nLinear evaluation')
    # res = {}
    # res['test_acc_1'], res['test_acc_bal_1']     = ut.linear_eval(model, train_loader_lin_1,   test_loader_lin, args, '  1%', True)
    # res['test_acc_10'], res['test_acc_bal_10']   = ut.linear_eval(model, train_loader_lin_10,  test_loader_lin, args, ' 10%', True)
    # res['test_acc_100'], res['test_acc_bal_100'] = ut.linear_eval(model, train_loader_lin_100, test_loader_lin, args, '100%', True)
    # faster alternative - that does the same thing
    train_inds = [np.array(train_inds_lin_1), np.array(train_inds_lin_10), np.arange(len(train_set_lin))]
    train_split_perc = [1, 10, 100]
    res = ut.linear_eval_all(model, train_loader_lin_100, test_loader_lin, args, train_inds, train_split_perc, True)
    
    if args['save_output']:
        op = {}
        op['args'] = args 
        op['epoch'] = args['epochs']
        op['results'] = res
            
        with open(args['op_res_name'], 'w') as da:
            json.dump(op, da, indent=2)
    
        op['state_dict'] = model.state_dict()
        torch.save(op, args['op_file_name'])
        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Self-Supervised with Context')

    parser.add_argument('--dataset', default='cct20', choices=['cct20', 'kenya', 'icct','serengeti','fmow'], type=str)
    parser.add_argument('--train_loss', default='simclr', 
                        choices=['simclr', 'triplet', 'triplet_hard', 'simsiam', 'rand_init', 'imagenet', 'supervised'], type=str)
    
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--not_cached_images', dest='cache_images', action='store_false')  # default for cache_images will be True
    parser.add_argument('--not_pretrained', dest='pretrained', action='store_false')  # default for pretrained will be True

    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--learning_rate_mult', default=0.03, type=float)
    parser.add_argument('--im_res', default=112, choices=[112, 224], type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--projection_dim', default=128, type=int)
    parser.add_argument('--supervised_amt', default=1, choices=[1, 10, 100], type=int)
    parser.add_argument('--seed', default=2001, type=int)
    
    parser.add_argument('--oracle_pos_noise_amt', default=0.0, type=float) 
    parser.add_argument('--oracle_pos_same_loc', dest='return_oracle_pos_same_loc', action='store_true')
    
    parser.add_argument('--pos_type', default='augment_self', 
                        choices=['augment_self', 'seq_positive', 'context_sample', 'image_emb_sample', 
                                 'context_and_image_emb_sample',, 'oracle_positive' ], type=str)
    parser.add_argument('--con_temp_select', default=0.05, type=float)
    parser.add_argument('--emb_temp_select', default=0.5, type=float)  
    parser.add_argument('--burn_in_select', default=1, type=int)  
    parser.add_argument('--pretrained_model', default='', type=str)
    parser.add_argument('--train_from_megadetector', action='store_true')
    parser.add_argument('--exp_name', default='', type=str)
    
    # turn the args into a dictionary
    args = vars(parser.parse_args())


    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    
    # CIFAR has not been tested with all the new additions 
    # Assume that is it not working
    # args['dataset'] = 'cifar_mnist' 
    # args['data_dir'] = 'data/'
    # args['batch_size'] = 512 
    # args['pretrained'] = False
    # args['eval_interval'] = 20
    # args['return_context'] = False
        
    args['data_dir'] = os.path.join('cam_data/', args['dataset'], '')
    args['metadata'] = os.path.join(args['data_dir'], args['dataset']+'_context_final.csv')
    if args['train_from_megadetector']:
        #args['metadata_md'] = os.path.join(args['data_dir'], args['dataset']+'_context_md_final.csv')
        #args['metadata_md'] = os.path.join(args['data_dir'], args['dataset']+'_context_md_extra_final.csv') ## double number of md images
        args['metadata_md'] = os.path.join(args['data_dir'], args['dataset']+'_context_md_extra_location_final.csv') ## double number of md images, but from test locations
    args['learning_rate'] = args['learning_rate_mult']*args['batch_size']/256
    args['momentum'] = 0.9
    args['weight_decay'] = 0.0005
    args['schedule'] = 'cosine'
    args['eval_interval'] = args['epochs']+1  # i.e. dont run eval during training  
    args['workers'] = 6
    args['burn_in'] = 0  # if > 0, the backbone will be frozen for "burn_in" epochs
    args['device'] = 'cuda'  # should use this consistently in code
    args['lin_max_iter'] = 1000  # number of iterations in the linear evaluation  
    
    args['triplet_margin'] = 0.3
    args['temperature'] = 0.5
            
    args['return_context'] = True
    args['return_alt_pos'] = False
    args['return_seq_pos'] = False 
    args['return_oracle_pos'] = False
    args['store_embeddings'] = False
        
    # setup how positive images are selected 
    if args['pos_type'] == 'augment_self':
        pass
    elif args['pos_type'] == 'seq_positive':
        args['return_seq_pos'] = True
    elif args['pos_type'] == 'oracle_positive':
        args['return_oracle_pos'] = True
    elif args['pos_type'] in ['context_sample', 'context_and_image_emb_sample']:
        args['store_embeddings'] = True
        args['return_alt_pos'] = True
        
    args['save_output'] = True
    args['op_dir'] = 'results/' 
    args['op_dir_mod'] = 'models/' 
    
    if not os.path.isdir(args['op_dir']):
        os.makedirs(args['op_dir'])
    if not os.path.isdir(args['op_dir_mod']):
        os.makedirs(args['op_dir_mod'])
        
    cur_time = datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    args['cur_time'] = cur_time
    op_str = args['dataset'] + '_' + args['backbone'] + '_' + args['train_loss'] + '_' + args['cur_time'] 
    args['op_file_name'] = args['op_dir_mod'] + op_str + '.pt'
    args['op_res_name'] = args['op_dir'] + op_str + '.json'
    args['op_im_name'] = args['op_dir'] + op_str + '_' + str(args['epochs']) + '.png'
    
    if args['train_loss'] == 'imagenet':
        args['epochs'] = 0
        args['pretrained'] = True
        args['cache_images'] = False
    
    if args['train_loss'] == 'rand_init':
        args['epochs'] = 0
        args['pretrained'] = False
        args['cache_images'] = False
    
    print('\n**********************************')
    print('Experiment :', args['exp_name'])
    print('Dataset    :', args['dataset'])
    print('Train loss :', args['train_loss'])
    print('Pos type   :', args['pos_type'])
    print('Backbone   :', args['backbone'])
    print('Pretrained :', args['pretrained'])
    print('Cached ims :', args['cache_images'])    
    print('Op file    :', args['op_res_name'])
    
    main(args)
