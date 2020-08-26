#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:03:59 2020

@author: edward

dataset for pytorchd dataloading

"""
import torch
import os, glob, itertools, random
from torch.utils.data import Dataset, DataLoader
import numpy as np
#from tqdm import tqdm
import pprint
pp = pprint.PrettyPrinter(indent=4)
from numba import jit

@jit(nopython=True)
def euclidean_distance(p1,p2):
    return np.linalg.norm(p1 - p2)
    #return (p1 - p2).norm(2)

@jit(nopython=True)
def compute_local_distances2(positions):

    num_nodes = positions.shape[0]
    distances = np.ones((num_nodes, num_nodes)) * 1000000
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            distances[i,j] = euclidean_distance(positions[i,:2], positions[j,:2]) # remove angle
    
    return distances

def compute_local_distances(positions):
    
    return torch.from_numpy(compute_local_distances2(positions.numpy())).float()
    
def euclidean_distance3(p1,p2):
    
    return (p1 - p2).norm(2)
def compute_local_distances3(positions):
    num_nodes = positions.size(0)
    distances = torch.ones(num_nodes, num_nodes) * 1000000
    
    for i,j in itertools.product(range(num_nodes), range(num_nodes)):
        #if adj_mat[i,j]: # switched of as probabilities are used during training
        distances[i,j] = euclidean_distance3(positions[i,:2], positions[j,:2]) # remove angle
    
    return distances

class GraphDataset(Dataset):
    def __init__(self, path, noise_level=0, data_limit=0, min_length=2, normalize=False, weight_mask=False):
        
        print('Loading dataset from path', path)
        self.normalize = normalize
        self.files = sorted(glob.glob(os.path.join(path, '*.pth')))
        if data_limit:
            assert data_limit <= len(self.files), (len(self.files), path)
            self.files = self.files[:data_limit]
        self.noise_level = noise_level
        self.graph_data = [torch.load(file) for file in self.files]
        self.weight_mask = weight_mask
        
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        #graph_data = torch.load(self.files[idx])
        
        target = random.choice(range(32)) # 
        graph_data = self.graph_data[idx]
        #target = idx % 32
        optimal_actions = graph_data['optimal_actions'][:, target]
        features = graph_data['features']
        #resnet_features = graph_data['resnet_features']
        resnet_features = graph_data.get('resnet_features', None)
        weights = graph_data.get('weight_mat', None)
        if weights is not None:
            weights = torch.min(weights[:, target], torch.ones_like(weights[:, target])*1000.0)
        positions = graph_data['positions']
        #local_distances = graph_data['local_distance_matrix']
        target_node = graph_data['features'][target].unsqueeze(0) # remove target position
        target_one_hot = torch.zeros(32)
        target_one_hot[target] = 1
        local_distances = compute_local_distances(positions) # recompute to remove high values
        target_distances = graph_data['global_distance_matrix'][:,target]
        if self.normalize:
            local_distances = local_distances / local_distances.max()
        
        sample =  {'positions': positions,
                'features': features ,
                'optimal_actions': optimal_actions,
                'linkages': graph_data['predicted_adj_mat'],
                'ground_linkages': graph_data['ground_adj_mat'],
                'target_node': target_node, 
                'target_one_hot': target_one_hot, 
                #'target_id': torch.Tensor([target]).long(), 
                'local_distances': local_distances,
                #'path_lengths': graph_data['path_length_mat'],
                'target_path_lengths': graph_data['path_length_mat'][target],
                'target_distances': target_distances
                } # TODO this should not include adjacency
         
        if weights is not None:
            if self.weight_mask:
                weights[:2] = 0.0
            sample['weights'] =  weights

            
        if resnet_features is not None:
            sample['resnet_features'] = resnet_features.squeeze(0)
            
        return sample
        
def plot_graph(graph_data):
    
    # draw verticies
    
    
    import matplotlib.pyplot as plt
    num_nodes = graph_data['positions'].size(0)
    positions = graph_data['positions'].numpy()
    
    adj_mat = graph_data['ground_linkages'].numpy()
    optimal_actions = graph_data['optimal_actions']
    # draw edges
    for i,j in itertools.product(range(num_nodes), range(num_nodes)):
        if adj_mat[i,j]:
            plt.plot([positions[i,0], positions[j,0]],
                     [positions[i,1], positions[j,1]],alpha=0.1, c='k')
        
    plt.scatter(positions[:,:1],
                positions[:,1:2])      
    
    
    start = random.choice(range(num_nodes))
    end = np.argmax(graph_data['target_one_hot'].numpy())
    
    plt.scatter(positions[start,0],
                positions[start,1], c='g', s=200)
    plt.scatter(positions[end,0],
                positions[end,1], c='r', s=200)
    
    current = start
    while current != end:
        next_ind = optimal_actions[current]
        plt.plot([positions[current,0], positions[next_ind,0]],
                 [positions[current,1], positions[next_ind,1]],alpha=0.5, c='m')
        
        
        current = next_ind
        

def old():
    path = 'data/graph_data3_distance_weights/train/'
    
    for stage in ['train', 'val']:
        dataset = GraphDataset(path.replace('train', stage), data_limit=10)
        x_min, y_min = 1000.0, 1000.0
        x_max, y_max = -1000.0, -1000.0
        
        for i in range(1):
            graph_data = dataset[i]
            
            distances = graph_data['local_distances']
            x_min = min(x_min, distances.min().item())
            x_max = max(x_max, distances.max().item())
            
        print(stage, x_min, x_max)
                

if __name__ == '__main__':
    path = 'data/graph_data5_distance_weights/train/'
    
    stats = {
        'train': {},
        'val': {}
        
        }
    
    
    
    for stage in ['train', 'val']:
        dataset = GraphDataset(path.replace('train', stage), data_limit=0, normalize=True)
        example = dataset[0]
        for key in example.keys():
            
            stats[stage][key + '_min'] = 10000
            stats[stage][key + '_max'] = -10000
            stats[stage][key + '_min_max'] = 10000
            stats[stage][key + '_max_min'] = -10000
        num_issues = 0
        print('#'*40)
        print(stage)
        for i in range(len(dataset)):
            example = dataset[i]
            for key in example.keys():
                if torch.any(example[key] != example[key]):
                    print('nans', key, i, dataset.files[i])
                    os.remove(dataset.files[i])
                    num_issues += 1
                stats[stage][key + '_min'] = min(stats[stage][key + '_min'], example[key].min().item())
                stats[stage][key + '_max'] = max(stats[stage][key + '_max'], example[key].max().item())
                stats[stage][key + '_min_max'] = min(stats[stage][key + '_min'], example[key].max().item())
                stats[stage][key + '_max_min'] = max(stats[stage][key + '_max'], example[key].min().item())
                      
        stats[stage]['num_issues'] = num_issues
    pp.pprint(stats)
            
            
            
            
            
        
        
    