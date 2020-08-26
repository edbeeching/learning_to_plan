#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 16:42:18 2020

@author: edward



launch_gmedt graph_train_hyp_001.sh '--data_path data/graph_data3_distance_weights --hidden_size 256 --batch_size 32 --max_grad_norm 2 --weight_decay 0.00001 --schedule 250 --n_steps 8 --data_limit 32'

"""
import argparse
import os, glob, itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from graph_dataset import GraphDataset
from layers import Flatten, init
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def cartesian_product(m1, m2):
    # create the cartestian product of all elements in m1 and m2
    assert m1.size(0) == m2.size(0)
    batch_size = m1.size(0)
    num_examples = m1.size(1)
    num_features = m1.size(2)
    m1 = m1.unsqueeze(2).repeat(1,1,1,num_examples)
    m2 = m2.unsqueeze(1).repeat(1,1,num_examples,1)    
    prod = torch.cat([m1.permute(0,1,3,2).contiguous().view(batch_size, -1, num_features), 
                      m2.view(batch_size, -1, num_features)],dim=2)    
    return prod


class MLP(nn.Module):
    def __init__(self, i,o,h, args):
        super(MLP, self).__init__()
        self.args=args
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
               constant_(x, 0.1), 1.0)
        
        if args.true_mlp:
            self.mlp = nn.Sequential(
                init_(nn.Linear(i,h)),
                #nn.LayerNorm(h),
                nn.ReLU(True),
                init_(nn.Linear(h,o)),
                
                )  
        else:
            self.mlp = init_(nn.Linear(i,o))            
        
    def forward(self, x):
        x = self.mlp(x)
        if self.args.mlp_relu:
            x = F.relu(x)
        
        return x
    
    
    
class BoundNetwork(nn.Module):
    def __init__(self, i,o,h, args):
        # uses gating to approximate the bound on the distance
        self.args = args
        super(BoundNetwork, self).__init__()
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
               constant_(x, 0.1), 1.0)
        
        
        self.linear1 = init_(nn.Linear(i, o))
        self.linear2 = init_(nn.Linear(i, o))
        #self.linear3 = init_(nn.Linear(h, o))
        
    def forward(self, x):
        y = self.linear1(x)
        if self.args.tanh_gating:
            z = torch.tanh(self.linear1(x))
        else:
            z = torch.sigmoid(self.linear1(x))
        #print((y*z).size())
        #return self.linear3(z*y)
        return z*y
        
class BoundNetwork2(nn.Module):
    def __init__(self, i,o):
        # uses gating to approximate the bound on the distance
        
        super(BoundNetwork2, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
               constant_(x, 0.1), 1.0)    
        
        self.linear1 = init_(nn.Linear(i, i))
        self.linear2 = init_(nn.Linear(i, o))

        
    def forward(self, x):
        #print(x.size())
        z = torch.sigmoid(self.linear1(x))
        y = z*x # contains distance
        
        return self.linear2(y)        
        

class GNN(nn.Module):
    def __init__(self, in_size, r_size, num_nodes=32, args=None): 
        super(GNN, self).__init__()
        
        self.args = args
        self.num_nodes = num_nodes
        self.r_size = r_size
        
        if args.new_bound_net:
            self.bound_init = BoundNetwork2(in_size , r_size)
            self.bound_update = BoundNetwork2((in_size + r_size) * 2 , r_size)     
        else:
            self.bound_init = BoundNetwork(in_size , r_size, r_size, args)
            self.bound_update = BoundNetwork((in_size + r_size) * 2 , r_size, (in_size + r_size) * 2, args)
        self.distribution_function = MLP(r_size, num_nodes, r_size, args)
        self.prod_mlp = MLP(r_size, r_size, r_size, args)
        self.r_init = nn.Parameter(torch.zeros(1, num_nodes, r_size))
        
        if args.bound_update:
            self.gru = nn.GRU(r_size, r_size, args.gru_depth, batch_first=True)
        else:    
            self.gru = nn.GRU((in_size + r_size) * 2, r_size, args.gru_depth, batch_first=True)
            
        if args.no_gru:
            if args.bound_update:
                self.no_gru_linear = MLP(r_size, r_size, r_size, args)
            else:
                self.no_gru_linear = MLP((in_size + r_size) * 2, r_size, r_size, args)
            
            
        self.distance_regressor = MLP(r_size, 1, r_size, args)
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    torch.nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    torch.nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                torch.nn.init.zeros_(param)      
        
        
    def forward(self, batch, device):
        if self.args.use_probs:
            node_core_features = torch.cat([batch['local_distances'], 
                                        batch['linkages'],
                                        batch['target_one_hot'].unsqueeze(2),
                                        torch.eye(self.num_nodes).unsqueeze(0).repeat(self.args.batch_size, 1, 1).to(device)], dim=2)
        else:
            # use adjacency
            node_core_features = torch.cat([batch['local_distances'], 
                                        batch['ground_linkages'].float(),
                                        batch['target_one_hot'].unsqueeze(2),
                                        torch.eye(self.num_nodes).unsqueeze(0).repeat(self.args.batch_size, 1, 1).to(device)], dim=2)
        if self.args.use_features:
            node_core_features =  torch.cat([node_core_features, batch['features']], dim=2)
            
        if self.args.use_positions:
            node_core_features =  torch.cat([node_core_features, batch['positions']], dim=2)
        
        if self.args.use_resnet_features:
            node_core_features =  torch.cat([node_core_features, batch['resnet_features']], dim=2)
        
        batch_size = batch['local_distances'].size(0)
        num_nodes = batch['local_distances'].size(1)

        r_prime = self.bound_init(node_core_features) # initialize the r_prime bound vector        
        
        if self.args.use_mscl:
            dists = []
        
        
        if self.args.store_hidden:
            h = None 
        
        for step in range(self.args.n_steps):
            x = torch.cat([r_prime, node_core_features], dim=2)
            
            batch_size, num_nodes, features = x.size()
            product = cartesian_product(x, x).view(batch_size*num_nodes, num_nodes, features*2)
            
            if self.args.bound_update:
                product = self.bound_update(
                    product.view(batch_size*num_nodes*num_nodes, features*2)).view(
                        batch_size*num_nodes, num_nodes, self.r_size)
            
            
            if self.args.no_gru:
                x = product.mean(1)
                x = self.no_gru_linear(x).view(x.size(0),1,-1)
            else:
                if self.args.store_hidden:
                    x,h = self.gru(product, h)
                else:
                    x,h = self.gru(product)
            
            r_prime = self.prod_mlp(x[:,-1]).view(batch_size, num_nodes, self.r_size)
            if self.args.use_mscl:
                dists.append(self.distribution_function(r_prime.view(batch_size * num_nodes, -1)))
        
        if self.args.use_mscl:
            return dists, None
        
        dist = self.distribution_function(r_prime.view(batch_size * num_nodes, -1))
        distances = self.distance_regressor(r_prime.view(batch_size * num_nodes, -1))
        return dist, distances


def create_encoding_matrix(num_nodes):
    # position encoding for cartesian product of graph networks
    mat = torch.eye(num_nodes).unsqueeze(0)
    
    return cartesian_product(mat, mat)


def accuracy(preds, targets):
    _, predicted = torch.max(preds.detach(), 1)
    correct = (predicted == targets).sum().item()    
    
    return correct / targets.size(0)

def paccuracy(preds, targets, target_path_lengths, min_length=2):
    # accuracy for path lengths of min length or greater
    _, predicted = torch.max(preds.detach(), 1)
    correct = (predicted == targets)
    
    mask = target_path_lengths >= min_length 
    mask = mask.view(-1)
    
    total = ((correct * mask).sum() / (mask.sum() + 1e-8)).item()

    return total


def step_wise_path_loss(path_lengths, n_steps):
    assert len(path_lengths.size()) == 2 # batch, path lengths for single target from each node
    masks = torch.zeros_like(path_lengths).view(-1, 1, path_lengths.size(1)).repeat(1, n_steps, 1)
    for i in range(n_steps):
        # substract 1 because paths of length one are found after one round of message passing
        masks[:, i] = (path_lengths <= (i-1)).float() 
        
    return masks.view(-1)
    

def epoch(model, loader, device, optimizer=None):
    losses = []
    accs = []
    paccs = [[] for _ in range(5)] #path accuracy
    
    for batch in loader:
        batch = {k:v.to(device) for k,v in batch.items()}
   
        preds, dists = model(batch, device)
        
        if USE_MSCL:
            loss = nn.CrossEntropyLoss(reduction='none')(torch.cat(preds, dim=1).view(-1, N_NODES), batch['optimal_actions'].long().view(-1).repeat(N_STEPS))
            if USE_WEIGHTS:
                loss = loss * batch['weights'].view(-1).repeat(N_STEPS)
            
            mask = step_wise_path_loss(batch['target_path_lengths'], N_STEPS)
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + 1e-6)
        
            preds = preds[-1]
          
        else:
            if USE_WEIGHTS:
                assert 'weights' in batch.keys()
                loss = nn.CrossEntropyLoss(reduction='none')(preds.view(-1, N_NODES), batch['optimal_actions'].long().view(-1))
                loss = (loss * batch['weights'].view(-1)).mean()
            else:
                loss = nn.CrossEntropyLoss()(preds.view(-1, N_NODES), batch['optimal_actions'].long().view(-1))
            
            if args.regress_distance:
                dist_loss = nn.MSELoss(reduction='none')(dists.view(-1), batch['target_distances'].view(-1))
                loss = loss + args.dist_weight * (dist_loss * batch['weights'].view(-1)).mean()
            
        
        acc = accuracy(preds.view(-1, N_NODES), batch['optimal_actions'].long().view(-1))
        accs.append(acc)
        
        
        for i in range(NUM_PACCS):
            pacc = paccuracy(preds.view(-1, N_NODES), batch['optimal_actions'].long().view(-1), batch['target_path_lengths'], min_length=i+2)
            paccs[i].append(pacc)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
        losses.append(loss.item())
        
    return sum(losses) / len(losses), sum(accs) / len(accs), [sum(p) / len(p) for p in paccs]
        


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--schedule", type=int, default=100)
    parser.add_argument("--data_limit", type=int, default=1000)
    parser.add_argument("--val_data_limit", type=int, default=1600)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--feature_size", type=int, default=512)
    parser.add_argument("--gru_depth", type=int, default=1)
    parser.add_argument("--n_steps", type=int, default=4)
    parser.add_argument("--val_divider", type=int, default=10)
    parser.add_argument("--num_paccs", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.00001)
    parser.add_argument("--use_weights", default=False, action='store_true')
    parser.add_argument("--use_probs", default=False, action='store_true')
    parser.add_argument("--use_features", default=False, action='store_true')
    parser.add_argument("--use_resnet_features", default=False, action='store_true')
    parser.add_argument("--use_positions", default=False, action='store_true')
    parser.add_argument("--mask_weights", default=False, action='store_true')
    parser.add_argument("--normalize", default=False, action='store_true')
    parser.add_argument("--use_mscl", default=False, action='store_true', help='use a multistep class loss')
    parser.add_argument("--save_model", default=False, action='store_true')
    parser.add_argument("--save_dir", default='tmp/')
    parser.add_argument("--data_path", default='data/graph_data3_distance_weights/train/', type=str)
    parser.add_argument("--load_checkpoint", default='', type=str)
    
    
    #other ideas
    
    parser.add_argument("--mlp_relu", default=False, action='store_true')
    parser.add_argument("--cosine_scheduler", default=False, action='store_true')
    parser.add_argument("--true_mlp", default=False, action='store_true')
    parser.add_argument("--tanh_gating", default=False, action='store_true')
    parser.add_argument("--mod_drop", default=False, action='store_true')
    parser.add_argument("--mod_decay", default=False, action='store_true')
    parser.add_argument("--store_hidden", default=False, action='store_true')
    parser.add_argument("--bound_update", default=False, action='store_true')
    parser.add_argument("--new_bound_net", default=False, action='store_true')
    parser.add_argument("--regress_distance", default=False, action='store_true')
    parser.add_argument("--dist_weight", default=1.0, type=float)
    parser.add_argument("--no_gru", default=False, action='store_true')
    
    

    return parser

if __name__ == '__main__':
    
    parser = parse_args()
    args = parser.parse_args()
    
    
    BATCH_SIZE = args.batch_size
    N_NODES = 32   
    DATA_LIMIT = args.data_limit
    R_SIZE = args.hidden_size
    MAX_GRAD_NORM = args.max_grad_norm
    LR = args.lr
    EPOCHS = args.epochs
    SCHEDULE = args.schedule
    FEATURE_SIZE = args.feature_size
    USE_PROBS = args.use_probs
    USE_FEATURES = args.use_features
    USE_RESNET_FEATURES = args.use_resnet_features
    USE_POSITIONS = args.use_positions
    USE_MSCL = args.use_mscl
    N_STEPS = args.n_steps
    USE_WEIGHTS = args.use_weights
    GRU_DEPTH = args.gru_depth
    NUM_PACCS = args.num_paccs
    MOD_DROP = args.mod_drop
    ORIG_MOD_PROB = 0.5 # decay to zero to use only to connection probabilities
    MOD_PROB = ORIG_MOD_PROB # decay to zero to use only to connection probabilities
    
    
    input_size = 97
    if MOD_DROP:
        input_size += N_NODES
    if USE_FEATURES:
        input_size += FEATURE_SIZE
    if USE_RESNET_FEATURES:
        input_size += FEATURE_SIZE
    if USE_POSITIONS:
        input_size += 4 # x,y, sin(theta), cos(theta)

    time_string = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    

    job_id = 12345
    job_id = os.environ.get('JOB_ID')
    if job_id is None:
        job_id = os.environ.get('SLURM_JOBID')
    writer = SummaryWriter(log_dir=f'runs/{time_string}_{job_id}_{BATCH_SIZE}_{DATA_LIMIT}_{R_SIZE}_{MAX_GRAD_NORM}_{LR}_{SCHEDULE}_{USE_PROBS}_{USE_FEATURES}_{USE_RESNET_FEATURES}_{USE_POSITIONS}_{USE_MSCL}')
    path = args.data_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset = GraphDataset(path, data_limit=DATA_LIMIT, normalize=args.normalize, weight_mask=args.mask_weights)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=True, drop_last=True)
    val_dataset = GraphDataset(path.replace('train', 'val'), data_limit=args.val_data_limit, normalize=args.normalize, weight_mask=args.mask_weights)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, shuffle=False, drop_last=True)
    
    model = GNN(input_size,  R_SIZE, args=args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True, weight_decay=args.weight_decay) # amsgrad fixes an error in Adam and is generally better (https://openreview.net/forum?id=ryQu7f-RZ)
    
    if args.cosine_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, SCHEDULE)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, SCHEDULE, gamma=0.1)
    
    print('Starting training')
    print('train size:', len(train_dataset))
    for k,v in sorted(args.__dict__.items()):
        print(k,v) 
    
    best_val = 0.0
    
    for e in range(EPOCHS):
        model.train()
        train_loss, train_acc, ptrain_acc = epoch(model, train_loader, device, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss, val_acc, pval_acc = epoch(model, val_loader, device)
        
        print('E:{:003} TRL:{:2.5f} VL:{:2.5f} TRA:{:2.3f} VA:{:2.3f}'.format(e, train_loss, val_loss, train_acc, val_acc))
        print(ptrain_acc, pval_acc)
        scheduler.step()
        
        writer.add_scalar('Loss/train', train_loss, e)
        writer.add_scalar('Loss/val', val_loss, e)
        writer.add_scalar('Acc/train', train_acc, e)
        writer.add_scalar('Acc/val', val_acc, e)
        
        for i in range(NUM_PACCS):
            writer.add_scalar('PAcc_{}/train'.format(i+2), ptrain_acc[i], e)
            writer.add_scalar('PAcc_{}/val'.format(i+2), pval_acc[i], e)

        if pval_acc[0] > best_val and args.save_model:
            # add mod drop decay criteria

            assert os.path.exists(args.save_dir), 'model save dir does not exist'   
            
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')

            state = model.state_dict()
            checkpoint = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'hyp': vars(args),
                    'epoch':e               
                }
            torch.save(checkpoint, checkpoint_path)
            best_val = pval_acc[0]

    writer.close()