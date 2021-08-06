import torch
import torch.nn.functional as F

def nt_xent(x1, x2, args):
    # assumes that the input data is stacked i.e. 
    # x1 (B1 C H W) + x2 (B2 C H W) - B1 typically == B2
    
    x = torch.cat((x1, x2), 0)
    x = F.normalize(x, dim=1)
    x_scores = x @ x.t()
    x_scale = x_scores / args['temperature']   # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5
    
    # targets 2N elements.
    if x1.shape[0] == 1:
        # last element is the target i.e. all should be the same
        targets = torch.zeros(x.shape[0], device=x.device).long() 
        x_scale[0,0] = 1.0 / args['temperature'] 
    else: 
        # data is stacked in two halves
        targets = torch.arange(x.shape[0], device=x.device).long()
        targets[:x.shape[0]//2] += x.shape[0]//2
        targets[x.shape[0]//2:] -= x.shape[0]//2
        
    return F.cross_entropy(x_scale, targets)


def triplet_loss(emb, args, dist_type='cosine', margin=0.3):
    # NOTE currently just randomly selects indices as negatives 

    b_size = emb.shape[0]
    inds = torch.randint(0, b_size, (b_size//2, ))
    mask = (inds != torch.arange(b_size//2)).float().cuda()            
    
    if dist_type == 'l2':
        loss = (mask*F.triplet_margin_loss(emb[:b_size//2, :], emb[b_size//2:, :], 
                                           emb[inds, :], margin=margin, reduction='none')).mean()
                                            
    elif dist_type == 'cosine':
        pos_dist = (-F.cosine_similarity(emb[:b_size//2, :], emb[b_size//2:, :], dim=1) + 1)/2
        neg_dist = (-F.cosine_similarity(emb[:b_size//2, :], emb[inds, :], dim=1) + 1)/2
        hinge_dist = torch.clamp(margin + pos_dist - neg_dist, min=0.0)
        loss = (mask*hinge_dist).mean()
    
    return loss
    
        
def triplet_hard_loss(emb, args, num_closest=2, margin=0.3):
    # choose randomly from the top num_closest as a negative
        
    b_size = emb.shape[0]
    emb_sn = F.normalize(emb[:b_size//2, :], dim=1)
    x_scores = emb_sn @ emb_sn.t()
    
    close_inds = torch.argsort(-x_scores, 1)[:, 1:]
    r_inds = torch.randint(0, num_closest, (b_size//2, )).cuda()
    inds = close_inds.gather(1, r_inds.unsqueeze(1))[:, 0]
        
    pos_dist = (-F.cosine_similarity(emb[:b_size//2, :], emb[b_size//2:, :], dim=1) + 1)/2
    neg_dist = (-F.cosine_similarity(emb[:b_size//2, :], emb[inds, :], dim=1) + 1)/2
    loss = torch.clamp(margin + pos_dist - neg_dist, min=0.0).mean()
    
    return loss
    
    
def simsiam(p1, z1, p2, z2, args):  
    # this does the l2 normalization internally
    l1 = -F.cosine_similarity(p1, z2.detach(), dim=-1).mean() / 2.0
    l2 = -F.cosine_similarity(p2, z1.detach(), dim=-1).mean() / 2.0    
    return l1 + l2
    
    


