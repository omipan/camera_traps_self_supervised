import matplotlib.pyplot as plt
import pandas as pd  
import numpy as np
from datasets import load_context
from scipy.spatial.distance import cdist

import torch
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader
import torch.nn.functional as F

import utils as ut
import datasets as ds


args = {}

# args['dataset'] = 'kenya'     
# args['data_dir'] = 'cam_data/kenya/'
# args['metadata'] = 'cam_data/kenya/kenya_context_final.csv'
# split_names = ['train']

args['dataset'] = 'cct20' 
args['data_dir'] = 'cam_data/cct20/'
args['metadata'] = 'cam_data/cct20/cct20_context_final.csv'
split_names = ['train_images']

# args['dataset'] = 'icct' 
# args['data_dir'] = 'cam_data/icct/'
# args['metadata'] = 'cam_data/icct/icct_context_final.csv'
# split_names = ['train_images']

da = pd.read_csv(args['metadata'])
da['prev_same_next_img_boxes'] = da['prev_same_next_img_boxes'].apply(lambda x: eval(x))

# only use the relevant split data
da = da[da['img_set'].isin(split_names)]

con_dict = load_context(da)
#context = torch.tensor(con_dict['con_standard'])
context = torch.tensor(con_dict['con_time_scaled'])
location = torch.tensor(con_dict['location_ids'])

_, targets_np = np.unique(da['category_id'].values, return_inverse=True) 
un_targets = np.unique(targets_np)
targets = torch.tensor(targets_np)
cls_dict = dict(zip(da['img_path'].values.tolist(), targets_np.tolist()))

print('dataset', args['dataset'])
print('context size', context.shape[0], context.shape[1])
print('num classes', len(un_targets))
print('avg neighs', round(np.mean([len(ff) for ff in da['prev_same_next_img_boxes'].values]), 3))

con_dist = torch.cdist(context, context)
con_dist = con_dist + (location != location.unsqueeze(1)).float()*1e5
#con_dist += torch.eye(con_dist.shape[0])*1e5
#con_dist = torch.softmax(-con_dist / 0.05, dim=1)
#inds = torch.argsort(con_dist, 1)
_, cinds = torch.topk(-con_dist, 10, 1)

# find percentage of neighbours that are the same class
print('Context similarity')
for ii in range(cinds.shape[1]):
    acc = (targets == targets[cinds[:, ii]]).float().mean().item()
    dist = con_dist[torch.arange(con_dist.shape[0]), cinds[:, ii]].mean().item()
    print(ii, '\t', round(acc, 3), '\t', round(dist, 3))


args2 = {}
args2['device'] = 'cuda'
args2['train_loss'] = 'supervised'
args2['backbone'] = 'resnet18'
args2['pretrained'] = True
args2['projection_dim'] = 1
args2['store_embeddings'] = False
args2['dataset'] = args['dataset']
args2['metadata'] = args['metadata']
args2['data_dir'] = args['data_dir']
args2['return_context'] = True
args2['oracle_pos_noise_amt'] = 0.0
args2['batch_size'] = 256
args2['im_res'] = 112
args2['workers'] = 6
args2['return_alt_pos'] = False
args2['return_seq_pos'] = False
args2['return_oracle_pos'] = False
args2['return_oracle_pos_same_loc'] = False
args2['cache_images'] = False

# Need to careful that the correct test transforms are used
_, train_set_lin, _, _, _ = ds.get_dataset(args2)
train_loader = DataLoader(train_set_lin, batch_size=args2['batch_size'], num_workers=args2['workers'], shuffle=False)
model = resnet18(pretrained=True).to(args2['device'])
model.fc = torch.nn.Identity()

print('Extracting features')
x_train, y_train, train_ids = ut.get_features(model, train_loader, args2, 'target', standard_backbone=True)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=128)
# x_train = pca.fit_transform(x_train)


emb = F.normalize(torch.tensor(x_train), dim=1)
emb_dist = emb@emb.t()
emb_dist = emb_dist# - (location != location.unsqueeze(1)).float()*1e5
_, einds = torch.topk(emb_dist, 10, 1) ## higher is better

assert (targets == torch.tensor(y_train)).float().mean()

print('Imagenet similarity')
for ii in range(einds.shape[1]):
    acc = (targets == targets[einds[:, ii]]).float().mean().item()
    dist = emb_dist[torch.arange(emb_dist.shape[0]), einds[:, ii]].mean().item()
    print(ii, '\t', round(acc, 3), '\t', round(dist, 3))


# # amount of times that same image sequence is the same
# seq_acc = np.zeros(da.shape[0])
# for ii, ff in enumerate(da['prev_same_next_img_boxes'].values.tolist()):
#     m_tar = cls_dict[da['img_path'].values[ii]]
#     tars = []
#     for gg in ff:
#         tars.append(cls_dict[gg])
# 
#     if len(ff) > 0:
#         seq_acc[ii] = (np.array(tars) == m_tar).mean()
#     else:
#         seq_acc[ii] = -1
# print(np.unique(seq_acc, return_counts=True))    

if False:
    tinds0, tinds1 = torch.triu_indices(con_dist.shape[0], con_dist.shape[1], 1)
    tar_mask = (targets == targets.unsqueeze(1))
    pos_inds = (tar_mask[tinds0, tinds1] == 1) 
    neg_inds = (tar_mask[tinds0, tinds1] == 0) 
    tinds0p = tinds0[pos_inds] 
    tinds1p = tinds1[pos_inds] 
    tinds0n = tinds0[neg_inds] 
    tinds1n = tinds1[neg_inds] 

    max_val = con_dist[tinds0p, tinds1p].max().item()

    plt.close('all')
    plt.figure(0)
    plt.hist(con_dist[tinds0p, tinds1p].numpy(), 1000, density=False)
    plt.xlim([0, max_val])
    #plt.ylim([0, 2.0])
    plt.savefig('pos_dist.png')

    plt.figure(1)
    plt.hist(con_dist[tinds0n, tinds1n].numpy(), 1000, density=False)
    plt.xlim([0, max_val])
    #plt.ylim([0, 2.0])
    plt.savefig('neg_dist.png')
