import torch
import numpy as np
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from torchvision.datasets import CIFAR10, MNIST
import pandas as pd

#from torchvision import transforms
#import transforms_mult as transforms
import torchvision.transforms as transforms

from calendar import monthrange
import math
import ast


def get_dataset(args):
        
    if args['dataset'] == 'cifar_mnist': 
        # 
        # NOTE this has not been updated in some time. 
        # 
        
        train_transform = transforms.Compose([transforms.RandomResizedCrop(32),
                                              transforms.RandomHorizontalFlip(p=0.5),
                                              get_color_distortion(s=0.5),
                                              transforms.ToTensor()])
        test_transform = transforms.ToTensor()
                                      
        train_set = CIFAR10Pair(root=args['data_dir'],
                                train=True,
                                transform=train_transform,
                                download=True) 
        modify_data_cifar_mnist(train_set, args, True)
        train_set.return_single_image = False
        train_set.num_examples = train_set.data.shape[0]
        train_set.num_classes = np.unique(train_set.targets).shape[0]
        train_set.context_size = 0 
        
        
        # data for linear evaluation
        train_set_lin = CIFAR10Pair(root=args['data_dir'], train=True, transform=test_transform, download=False)
        train_set_lin.return_single_image = True
        modify_data_cifar_mnist(train_set_lin, args, True)
        
        # random subset
        rnd = np.random.RandomState(args['seed'])
        train_set_lin.perc_1_inds  = rnd.choice(train_set.num_examples, int(train_set.num_examples*0.01), replace=False).tolist()        
        train_set_lin.perc_10_inds = rnd.choice(train_set.num_examples, int(train_set.num_examples*0.1), replace=False).tolist()        
        
        test_set_lin = CIFAR10Pair(root=args['data_dir'], train=False, transform=test_transform, download=False)
        test_set_lin.return_single_image = True
        modify_data_cifar_mnist(test_set_lin, args, False)
        
        if args['train_loss'] == 'simclr_seq_pos':
            pritn('\n\nWarning. simclr_seq_pos is not implmented for cifar_mnist.')
    

    elif args['dataset'] == 'cct20':  
        train_transform, test_transform = im_transforms(args)
        train_set = IMAGE_DATASET(args, train_transform, ['train_images', 'trans_val_images', 'cis_val_images'], 
                                 args['return_alt_pos'], args['return_seq_pos'], args['return_oracle_pos'], 
                                  False, args['cache_images'])

        

        # ['train_images', 'trans_val_images', 'cis_val_images']
        train_set_lin = IMAGE_DATASET(args, test_transform, ['train_images'], 
                                       False, False, False, True, False)
        test_set_lin = IMAGE_DATASET(args, test_transform, ['trans_test_images', 'cis_test_images'], 
                                      False, False, False, True, False)

    elif args['dataset'] == 'icct':  
        train_transform, test_transform = im_transforms(args)
        train_set = IMAGE_DATASET(args, train_transform, ['train_images', 'trans_val_images', 'cis_val_images'], 
                                   args['return_alt_pos'], args['return_seq_pos'], args['return_oracle_pos'], 
                                  False, args['cache_images'])

        # ['train_images', 'trans_val_images', 'cis_val_images']
        train_set_lin = IMAGE_DATASET(args, test_transform, ['train_images'], 
                                      False, False, False, True, False)
        test_set_lin = IMAGE_DATASET(args, test_transform, ['trans_test_images', 'cis_test_images'], 
                                      False, False, False, True, False)

    elif args['dataset'] == 'serengeti':  
        train_transform, test_transform = im_transforms(args)
        train_set = IMAGE_DATASET(args, train_transform, ['train'], args['return_alt_pos'], 
                                  args['return_seq_pos'], args['return_oracle_pos'], False, args['cache_images'])

        train_set_lin = IMAGE_DATASET(args, test_transform, ['train'], False, False, False, True, False)
        test_set_lin = IMAGE_DATASET(args, test_transform, ['test'], False, False, False, True, False)
    
    elif args['dataset'] == 'kenya':  
        train_transform, test_transform = im_transforms(args)
        train_set = IMAGE_DATASET(args, train_transform, ['train', 'val'], args['return_alt_pos'], 
                                  args['return_seq_pos'], args['return_oracle_pos'], False, args['cache_images'],args['train_from_megadetector'])

        # ['train', 'val']
        train_set_lin = IMAGE_DATASET(args, test_transform, ['train'], False, False, False, True, False)
        test_set_lin = IMAGE_DATASET(args, test_transform, ['test'], False, False, False, True, False)

    elif args['dataset'] == 'fmow':  
        train_transform, test_transform = im_transforms(args)
        train_set = IMAGE_DATASET(args, train_transform, ['train'], args['return_alt_pos'], 
                                  args['return_seq_pos'], args['return_oracle_pos'], False, args['cache_images'])

        # ['train', 'val']
        train_set_lin = IMAGE_DATASET(args, test_transform, ['train'], False, False, False, True, False)
        test_set_lin = IMAGE_DATASET(args, test_transform, ['test'], False, False, False, True, False)

            
    return train_set, train_set_lin, test_set_lin, train_set_lin.perc_1_inds, train_set_lin.perc_10_inds
    
     
def im_transforms(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args['im_res'], scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        get_color_distortion(s=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std= [0.229, 0.224, 0.225]) # Imagenet means and stds
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((args['im_res'], args['im_res'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std= [0.229, 0.224, 0.225]) # Imagenet means and stds
    ])
    return train_transform, test_transform
    

def modify_data_cifar_mnist(cifar, args, is_train): 
    # modifies the data in place so that we use cifar as background and mnist as FG
    mnist = MNIST(root=args['data_dir'], train=is_train, download=True)
    
    rnd = np.random.RandomState(args['seed'])
    inds = torch.tensor(rnd.choice(mnist.data.shape[0], cifar.data.shape[0], replace=False))
    mnist_ims = mnist.data[inds, :].unsqueeze(-1).repeat([1,1,1,3]).numpy()
    
    # combine the "foreground" mnist with the "background" cifar
    mask = mnist_ims.astype(np.float32) / 255.0
    comb_data = cifar.data[:, 2:-2, 2:-2, :].astype(np.float32)*(1.0-mask) + mnist_ims.astype(np.float32)*(mask)
    # comb_data = cifar.data[:, 2:-2, 2:-2, :].astype(np.float32) + mnist_ims.astype(np.float32)
    cifar.data[:, 2:-2, 2:-2, :] = np.clip(comb_data, 0, 255).astype(np.uint8)
    cifar.targets = mnist.targets[inds].numpy().tolist()
    

# color distortion composed by color jittering and color dropping.
# See Section A of SimCLR: https://arxiv.org/abs/2002.05709
def get_color_distortion(s=0.5):  # 0.5 for CIFAR10 by default
    # s is the strength of color distortion
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort


class IMAGE_DATASET(torch.utils.data.Dataset):

    def __init__(self, args, transform, split_names, return_alt_pos, return_seq_pos, 
                 return_oracle_pos, return_single_image, cache_images,train_from_megadetector=False): 
        
        
        if train_from_megadetector:
            print('here')
            da = pd.read_csv(args['metadata_md'],converters={'prev_same_next_img_boxes': ast.literal_eval})
            da['im_id'] = np.arange(da.shape[0])
            un_targets, targets = np.unique(da['category_id'].values, return_inverse=True) 
            da['category_id_un'] = targets
            self.num_classes = 20
        else:
            da = pd.read_csv(args['metadata'],converters={'prev_same_next_img_boxes': ast.literal_eval})
            da['im_id'] = np.arange(da.shape[0])
            # get the class labels before some of the data is excluded
            un_targets, targets = np.unique(da['category_id'].values, return_inverse=True) 
            da['category_id_un'] = targets
            self.num_classes = un_targets.shape[0]
        
        # load the context data
        context_dict = load_context(da)

        # only use the relevant split's data
        da = da[da['img_set'].isin(split_names)]

        #da['prev_same_next_img_boxes'] = da['prev_same_next_img_boxes'].apply(lambda x: x))
        
        inds_to_keep = da['im_id'].values
        self.context = torch.tensor(context_dict['con_standard'][inds_to_keep, :])
        self.location_id = context_dict['location_ids'][inds_to_keep]
        self.hour = context_dict['hour'][inds_to_keep]
        self.context_size = self.context.shape[1]
        
        self.context_dict = context_dict
        for kk in ['con_time', 'con_time_scaled', 'con_bbox']:
            self.context_dict[kk] = torch.tensor(self.context_dict[kk][inds_to_keep, :])
        
        self.return_context = args['return_context']
                
        self.perc_1_inds = np.where(da['is_in_train_1perc'].values)[0].tolist()
        self.perc_10_inds = np.where(da['is_in_train_10perc'].values)[0].tolist()

        self.return_alt_pos = return_alt_pos
        self.return_seq_pos = return_seq_pos
        self.return_oracle_pos = return_oracle_pos
        self.oracle_pos_noise_amt = args['oracle_pos_noise_amt'] 
        self.return_oracle_pos_same_loc = args['return_oracle_pos_same_loc']
        self.return_single_image = return_single_image    
        
        self.transform = transform
        self.data_root = args['data_dir']
    
        self.targets = da['category_id_un'].values  # keep as np array
        self.targets_orig = da['category_id'].values  # keep as np array
        self.im_paths = da['img_path'].values.tolist()

        self.seq_paths = da['prev_same_next_img_boxes'].values.tolist()
        self.alt_paths = [im for im in self.im_paths]  # just initialize as a deep copy
        self.num_examples = len(self.im_paths)

                
        # cache the image data in RAM
        # this will use a lot of memory and only makes sense for smallish datasets
        self.cache_images = cache_images
        self.im_cache = {}
        if self.cache_images:
            print('caching images ...')
            for pp in self.im_paths:
                
                self.im_cache[pp] = loader(self.data_root + pp)                  
                      
        
            print('caching images done\n')
                    
    def __len__(self):
        return len(self.im_paths)
        
    
    def get_image(self, root_dir, im_path):  
        if self.cache_images and im_path in self.im_cache:
            return self.im_cache[im_path].copy()
        else:
            return loader(root_dir+im_path)
    
    
    def update_alternative_positives(self, inds):
        for ii, new_ind in enumerate(inds):
            self.alt_paths[ii] = self.im_paths[new_ind]
        
    def __getitem__(self, idx):    
        op = {}
        op['target'] = self.targets[idx] 
        op['target_orig'] = self.targets_orig[idx] 
        op['location_id'] = self.location_id[idx] 
        op['hour'] = self.hour[idx] 
        op['id'] = idx
        
        if self.return_context:
            op['con'] = self.context[idx, :]
            
        img1_path = self.im_paths[idx]
        img1 = self.get_image(self.data_root, img1_path)        
        if self.return_single_image:                        
            op['im'] = self.transform(img1)
                
        else:                                 
            if self.return_seq_pos:
                # choose an image from the same "sequence", Note this could still select the same one as img1
                op['im_t1'] = self.transform(img1) 
                img2_path = np.random.choice([self.im_paths[idx]] + self.seq_paths[idx])
                img2 = self.get_image(self.data_root, img2_path)
                op['im_t2'] = self.transform(img2)
            
            elif self.return_alt_pos: 
                # the alt_paths list will be populated periodically (e.g. every epoch) for each image 
                op['im_t1'] = self.transform(img1)
                img2_path = self.alt_paths[idx]
                img2 = self.get_image(self.data_root, img2_path)
                op['im_t2'] = self.transform(img2)
                
            elif self.return_oracle_pos: 
                # if oracle_pos_noise_amt == 0 there will be no noise
                op['im_t1'] = self.transform(img1)
                if np.random.rand() >= self.oracle_pos_noise_amt:
                    inds_to_select_from = np.where(self.targets == self.targets[idx])[0]
                else:
                    inds_to_select_from = np.where(self.targets != self.targets[idx])[0]  
                    
                # only selects positves from the same location
                if self.return_oracle_pos_same_loc:
                    loc_inds = np.where(self.location_id[inds_to_select_from] == self.location_id[idx])[0]
                    inds_to_select_from = inds_to_select_from[loc_inds]
                    
                idx2 = np.random.choice(inds_to_select_from)
                img2_path = self.im_paths[idx2]
                img2 = self.get_image(self.data_root, img2_path)
                op['im_t2'] = self.transform(img2)
        
            else: 
                # augment_self i.e. two different augmentations of the same image 
                img2_path = img1_path

                op['im_t1'] = self.transform(img1)    
                op['im_t2'] = self.transform(img1)
        
        return op


def loader(im_path_full):
    with open(im_path_full, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')     
        
        
class CIFAR10Pair(CIFAR10):
    """Generate mini-batch pairs on CIFAR10 training set."""
    def __getitem__(self, idx): 

        op = {}
        op['target'] = self.targets[idx] 
        op['id'] = idx

        img = Image.fromarray(self.data[idx])
        if self.return_single_image:            
            op['im'] = self.transform(img)
                
        else:             
            op['im_t1'] = self.transform(img)    
            op['im_t2'] = self.transform(img)
                        
        return op

def cos_sin_encode(x): 
    #assume betwen 0 and 1
    op = np.zeros((x.shape[0], 2), dtype=np.float32)
    op[:, 0] = np.sin(math.pi*((2*x)-1))
    op[:, 1] = np.cos(math.pi*((2*x)-1))
    op = (op+1)/2.0
    return op
    
def load_context(da, return_box_info=False, return_loc_info=False):

    width = da['width'].values / da['img_width'].values
    height = da['height'].values / da['img_height'].values
    x1 = (da['x1'].values / da['img_width'].values) + (width/2)
    y1 = (da['y1'].values / da['img_height'].values) + (height/2)
    area = (da['width'].values*da['height'].values) / (da['img_width'].values*da['img_height'].values)  
    un_locs, loc_inds = np.unique(da['location'].values, return_inverse=True)
    loc = np.zeros((loc_inds.shape[0], un_locs.shape[0]))
    loc[np.arange(loc.shape[0]), loc_inds] = 1.0  
    more_than_one = (da['boxes_per_img_id'].values>1).astype(np.float32)

    # get count of number of days per year - choose a leap year
    num_days = np.cumsum([0] + [monthrange(2020, ii)[1] for ii in range(1, 13)])
    
    # assuming 24 hour time 
    tm1d = np.zeros(da.shape[0])
    hr1d = pd.to_datetime(da['datetime']).dt.hour.values.astype(np.float32)
    min1d = pd.to_datetime(da['datetime']).dt.minute.values.astype(np.float32)
    sec1d = pd.to_datetime(da['datetime']).dt.second.values.astype(np.float32)
    year = pd.to_datetime(da['datetime']).dt.year.values.astype(np.float32)
    month = pd.to_datetime(da['datetime']).dt.month.values - 1
    day1d = pd.to_datetime(da['datetime']).dt.day.values.astype(np.float32) - 1
    for ii in range(day1d.shape[0]):
        day1d[ii] = num_days[month[ii]] + day1d[ii]
    
    if np.unique(year).shape[0] == 1:
        year = np.zeros(da.shape[0])
    else:
        year -= year.min()
        year /= year.max()
        
    # tm1d = sec1d + (min1d*60) + (hr1d*60*60) + (day1d*24*60*60) + (year*366*24*60*60)
    # tm1d -= tm1d.min()
    # tm1d /= 10#tm1d.max()
    # tm1d = tm1d[..., np.newaxis]

    day1d /= 366    
    hr1d /= 23.0
    min1d /= 59.0
    sec1d /= 59.0
    
    # day1d -= day1d.min()
    # day1d /= day1d.max()
    
    day = cos_sin_encode(day1d)
    hr = cos_sin_encode(hr1d)
    mi = cos_sin_encode(min1d)
    sec = cos_sin_encode(sec1d)
    
    # day = day1d[..., np.newaxis]
    # hr = hr1d[..., np.newaxis]
    # mi = min1d[..., np.newaxis]
    # sec = sec1d[..., np.newaxis]
    
    year = year[..., np.newaxis]
    x1 = x1[..., np.newaxis]
    y1 = y1[..., np.newaxis]
    width = width[..., np.newaxis]
    height = height[..., np.newaxis]
    area = area[..., np.newaxis]   
    more_than_one = more_than_one[..., np.newaxis]
                        
    # from sklearn.cluster import KMeans
    # n_bb_clusters = 512
    # kmeans = KMeans(n_clusters=n_bb_clusters) 
    # kmeans.fit(np.hstack((x1, y1, width, height)))
    # bb_clust = np.zeros((x1.shape[0], n_bb_clusters))
    # bb_clust[np.arange(x1.shape[0]), kmeans.labels_] = 1.0 

    # from sklearn.cluster import KMeans
    # n_area_clusters = 256
    # kmeans = KMeans(n_clusters=n_area_clusters) 
    # kmeans.fit(np.hstack((x1, y1)))
    # area_clust = np.zeros((area.shape[0], n_area_clusters))
    # area_clust[np.arange(area.shape[0]), kmeans.labels_] = 1.0 
    
    op = {}
    op['con_time'] = np.hstack((year, day, hr, mi, sec)).astype(np.float32)
    op['con_time_scaled'] = np.hstack((year*100.0, day*10.0, hr*1.0, mi*0.1, sec*0.01)).astype(np.float32)
    op['con_bbox'] = np.hstack((x1, y1, width, height, more_than_one)).astype(np.float32)
    op['con_loc_onehot'] = loc.astype(np.float32)
    op['con_standard'] = np.hstack((op['con_time'], op['con_loc_onehot']))
    op['location_ids'] = loc_inds
    op['hour'] = (hr1d*23).astype(np.int)
    
    #context = np.hstack((year*100.0, day*10.0, hr*1.0, mi*0.1, sec*0.01, loc*100.0))
    #context = np.hstack((tm1d, loc*100000))

    return op
