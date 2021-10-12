from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
import torch
import matplotlib.pyplot as plt 
import cv2

def train_linear(x_train_ip, y_train, x_test_ip, y_test, max_iter, grid_search):
    
    x_train = x_train_ip.astype(np.float32).copy()
    x_test = x_test_ip.astype(np.float32).copy()
    
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    x_train = normalize(x_train, norm='l2')
    x_test = normalize(x_test, norm='l2')

    rseed = 0    
    if grid_search:
        parameters = {'C' : [0.001, 0.01, 0.1, 1, 10, 100]}
        #cls = LinearSVC(random_state=0, tol=1e-4, C=1., dual=False, max_iter=1000)
        cls = LogisticRegression(random_state=rseed, tol=1e-4, multi_class='multinomial', C=1., dual=False, max_iter=max_iter)
        #clf = GridSearchCV(cls, parameters, n_jobs=-1, cv=3, refit=True)        
        clf = GridSearchCV(cls, parameters, n_jobs=-1, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=rseed), refit=True)        
        clf.fit(x_train, y_train)
    else:
        #clf = LinearSVC(random_state=0, tol=1e-4, C=1., dual=False, max_iter=1000).fit(x_train, y_train)
        clf = LogisticRegression(random_state=rseed, tol=1e-4, multi_class='multinomial', C=1., dual=False, max_iter=max_iter, n_jobs=-1).fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)*100
    bal_acc = balanced_accuracy_score(y_test, y_pred)*100

    return acc, bal_acc


def get_features(model, loader, args, target_type, op_type='feat', standard_backbone=False): 
    # target_type == 'target' for class labels
    
    # extract features from the model 
    if op_type == 'feat':
        only_feats = True
    else:
        only_feats = False 
        
    model.eval()
    features = []
    targets = []    
    ids = []
    with torch.no_grad():
        for data in loader:
            data['im'] = data['im'].to(args['device'])
            
            if standard_backbone:
                features.append(model(data['im']).data.cpu().numpy())
            else:
                op = model(data['im'], only_feats=only_feats)
                features.append(op[op_type].data.cpu().numpy())

            targets.append(data[target_type].cpu().numpy())
            ids.append(data['id'].cpu().numpy())
    
    return np.vstack(features), np.hstack(targets), np.hstack(ids)
    
    
def linear_eval(model, train_loader, test_loader, args, amt, grid_search=False, target_type='target'):

    # extract train and test features
    x_train, y_train, ids_train = get_features(model, train_loader, args, target_type)
    x_test, y_test, ids_test = get_features(model, test_loader, args, target_type)
    
    # double checking that the order is correct - if not make sure train_loader/test_loader is not  shuffled
    # maybe also set num_workers == 0
    assert (ids_train == np.arange(ids_train.shape[0])).mean()
    assert (ids_test == np.arange(ids_test.shape[0])).mean()
    
    # make sure the labels are consistent and range from 0 to C-1  
    # LinearSVC can handle non-consecutive class labels at train time  
    # Be careful as it will be hard to compare class performance over different splits 
    _, inv_labels = np.unique(np.hstack((y_train, y_test)), return_inverse=True)
    y_train = inv_labels[:y_train.shape[0]]
    y_test = inv_labels[y_train.shape[0]:]

    # perform linear evaluation
    test_acc, test_acc_bal = train_linear(x_train, y_train, x_test, y_test, args['lin_max_iter'], grid_search)
    print('Linear eval ' + amt + ': acc {:.2f},  bal acc {:.2f}'.format(test_acc, test_acc_bal))
        
    return test_acc, test_acc_bal
    

def linear_eval_all(model, train_loader, test_loader, args, inds, amts, grid_search=False, target_type='target'):

    # extract train and test features - only do this once
    x_train_o, y_train_o, ids_train = get_features(model, train_loader, args, target_type)
    x_test_o, y_test_o, ids_test = get_features(model, test_loader, args, target_type)

    # loop over the different data splits 
    res = {}
    for ii in range(len(inds)): 
        # select subset of data
        x_train = x_train_o[inds[ii], :]
        y_train = y_train_o[inds[ii]]

        # make sure the labels are consistent and range from 0 to C-1   
        _, inv_labels = np.unique(np.hstack((y_train, y_test_o)), return_inverse=True)
        y_train = inv_labels[:y_train.shape[0]]
        y_test = inv_labels[y_train.shape[0]:]
                
        # perform linear evaluation
        test_acc, test_acc_bal = train_linear(x_train, y_train, x_test_o, y_test, args['lin_max_iter'], grid_search) 
        amt = str(amts[ii])
        res['test_acc_' + amt] = test_acc
        res['test_acc_bal_' + amt] = test_acc_bal
        print('Linear eval ' + (amt+'%').rjust(4) + ': acc {:.2f},  bal acc {:.2f}'.format(test_acc, test_acc_bal))
            
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def plot_progress(res, args):
    res = np.array(res)
    epochs = res[:, 0]
    accs = res[:, 1]
    accs_bal = res[:, 2]
    
    plt.close('all')
    plt.figure(0)
    plt.plot(epochs, accs, label='acc')
    plt.plot(epochs, accs_bal, label='bal acc')
    plt.grid(True)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title(args['dataset'] + ' - ' + args['train_loss'])
    plt.ylim(0, 100)
    plt.xlim(1, args['epochs'] + 1)
    plt.savefig(args['op_im_name'])

### Camera trap preprocessing utils
def modify_polygon_coords(coco_dataset):
    dataset = copy.deepcopy(coco_dataset)
    for ann_id in dataset.anns:

        ann_object = dataset.anns[ann_id]
        if 'bbox' in ann_object:
            img_object = dataset.imgs[ann_object['image_id']]
            #if MD
            #new_seg = convert_polygon_ratio_to_dim(ann["segmentation"][0],img['width'],img['height'])
            #ann["segmentation"] = [new_seg]
            new_box = convert_ratio_to_dim(ann_object["bbox"], img_object['width'], img_object['height'])
            ann_object['area'] = new_box[2] * new_box[3]
            ann_object["bbox"] = new_box
        else:
            ann_object['area'] = 0
            ann_object["bbox"] = [0,0,0,0]

    return dataset


def image_load_cv2(path,
                   convert_to_rgb=False):
    '''
    Load image and optionally convert to rgb as cv2 reads in bgr
    Args:
        path: image filepath
        convert_to_rgb: Boolean defining if we want conversion to RGB
    Returns:
        Loaded image
    '''

    # load image and convert to rgb as cv2 reads in bgr
    try:
        img = cv2.imread(path)
        if convert_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as ex:
        print('Tried to load and convert image from path {}'.format(path))
        template = "An exception of type {0} occurred for. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        print(message)
        return None