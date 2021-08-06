import torch.nn as nn
import torch
import torch.nn.functional as F


class EmbModel(nn.Module):
    
    def __init__(self, base_encoder, args):
        super().__init__()
    
        self.enc = base_encoder(pretrained=args['pretrained'])
        self.feature_dim = self.enc.fc.in_features
        self.projection_dim = args['projection_dim'] 
        self.proj_hidden = 512

        self.simsiam = False
        self.embed_context = False
        

        if args['dataset'] == 'cifar_mnist':
            # Customize for CIFAR10. Replace conv 7x7 with conv 3x3, and remove first max pooling.
            # See Section B.9 of SimCLR paper.
            self.enc.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.enc.maxpool = nn.Identity()

        # remove final fully connected layer of the backbone
        self.enc.fc = nn.Identity()  

        if args['store_embeddings']:
            self.emb_memory = torch.zeros(args['num_train'], args['projection_dim'], 
                                          requires_grad=False, device=args['device'])
            
        if args['train_loss'] == 'simsiam': 
            self.simsiam = True
            # combination of standard simclr projector with BN and simiam predictor
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
                                           nn.BatchNorm1d(self.proj_hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.proj_hidden, self.projection_dim)) 
            self.predictor = PredictionMLP(self.projection_dim, self.projection_dim//2, self.projection_dim)
        
        else:
            
            # standard simclr projector
            self.projector = nn.Sequential(nn.Linear(self.feature_dim, self.proj_hidden),
                                           nn.ReLU(),
                                           nn.Linear(self.proj_hidden, self.projection_dim)) 
            
        
    def update_memory(self, inds, x):
        m = 0.9
        with torch.no_grad():
            self.emb_memory[inds] = m*self.emb_memory[inds] + (1.0-m)*F.normalize(x.detach().clone(), dim=1, p=2)
            self.emb_memory[inds] = F.normalize(self.emb_memory[inds], dim=1, p=2)        
        
    def forward(self, x, only_feats=False, context=None):
        op = {}
        op['feat'] = self.enc(x) 
        
        if not only_feats:
        
            if self.simsiam:
                op['emb'] = self.projector(op['feat'])
                op['emb_p'] = self.predictor(op['emb'])
        
            else:
                op['emb'] = self.projector(op['feat'])

        return op

class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x
