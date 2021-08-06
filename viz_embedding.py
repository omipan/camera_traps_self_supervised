import numpy as np
import os
#from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt

import torch
from torchvision.models import resnet18, resnet50
from torch.utils.data import DataLoader

import utils as ut
import datasets as ds
from models import EmbModel


model_path = 'models/kenya_resnet18_simclr_2021_03_14__04_20_21.pt' 
op_dir = 'ims/'
op_format = 'png'  # png or pdf
op_file_root = op_dir + os.path.basename(model_path)[:-3] + '_'

if not os.path.isdir(op_dir):
    os.makedirs(op_dir)


print('Loading', model_path)
checkpoint = torch.load(model_path)
args = checkpoint['args']
args['cache_images'] = False

# NOTE: if there is an error loading any older models due to missings args
# just set them here. They should not matter as we are just using the backbone
# to load features

base_encoder = eval(args['backbone'])
model = EmbModel(base_encoder, args).to(args['device'])
msg = model.load_state_dict(checkpoint['state_dict'], strict=True)
print(msg)

_, train_set_lin, test_set_lin, _, _ = ds.get_dataset(args)
train_loader = DataLoader(train_set_lin, batch_size=args['batch_size'], 
                          num_workers=args['workers'], shuffle=False)
test_loader  = DataLoader(test_set_lin, batch_size=args['batch_size'], 
                          num_workers=args['workers'], shuffle=False)    


print('Extracting features')
# Note that the class labels here are the original ones from the csv file so they will not be
# sequential
x_train, y_train_orig, train_ids = ut.get_features(model, train_loader, args, 'target_orig')
x_test, y_test_orig, test_ids = ut.get_features(model, test_loader, args, 'target_orig')
_, y_train = np.unique(y_train_orig, return_inverse=True) 
_, y_test = np.unique(y_test_orig, return_inverse=True) 


print('Performing Dimensionality Reduction') 
# easy to switch this to TSNE, just replace the call to UMAP with TSNE
x_train_emb = UMAP(n_components=2).fit_transform(x_train)
x_test_emb = UMAP(n_components=2).fit_transform(x_test)
comb_emb = UMAP(n_components=2).fit_transform(np.vstack((x_train, x_test)))
train_con_emb = UMAP(n_components=2).fit_transform(train_set_lin.context)


print('Saving images to', op_file_root)
plt.close('all')
plt.figure(0, figsize=(10,10)) 
plt.scatter(x_train_emb[:, 0], x_train_emb[:, 1], s=5, c=y_train, cmap='tab20')
plt.title('Train 2D Embed')
plt.savefig(op_file_root + 'train_emb.' + op_format)

plt.figure(1, figsize=(10,10)) 
plt.scatter(x_test_emb[:, 0], x_test_emb[:, 1], s=5, c=y_test, cmap='tab20')
plt.title('Test 2D Embed')
plt.savefig(op_file_root + 'test_emb.' + op_format) 

plt.figure(2, figsize=(10,10)) 
comb_labels = np.ones(comb_emb.shape[0])
comb_labels[x_train_emb.shape[0]:] = 0
plt.scatter(comb_emb[:x_train_emb.shape[0], 0], comb_emb[:x_train_emb.shape[0], 1], label='train', s=1, cmap='Set1')
plt.scatter(comb_emb[x_train_emb.shape[0]:, 0], comb_emb[x_train_emb.shape[0]:, 1], label='test', s=1, cmap='Set1')
plt.legend()
plt.title('Train & Test 2D Embed')
plt.savefig(op_file_root + 'train_test_emb.' + op_format) 

plt.figure(3, figsize=(10, 10)) 
plt.scatter(train_con_emb[:, 0], train_con_emb[:, 1], s=5, c=y_train, cmap='tab20')
plt.title('Train Context 2D Embed')
plt.savefig(op_file_root + 'train_con_emb.' + op_format) 
