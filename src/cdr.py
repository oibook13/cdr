import sys,os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.datasets as datasets
import torch.utils.data as data


class Centroid(nn.Module):
    """
        Centroid in CDR
    """

    def __init__(self, 
                    in_features, 
                    normalized_radius=1.0, 
                    p=None, 
                    t=None):
        """
            Constructor for Centroid
            Input
                in_features: input feature dimension
                normalized_radius: the radius used in normalization for initialization
                p: a set of Lp distance metrics
                t: a set of temperatures w.r.t p
        """
        super(Centroid, self).__init__()
        basis = torch.rand(1, in_features)
        basis /= torch.norm(basis, p=2)
        basis *= normalized_radius
        self.basis = torch.nn.Parameter(basis)
        self.p = p
        self.T = t

    def forward(self,x):
        batch_basis = self.basis.expand(x.size(0), self.basis.size(1))
        x_d = None
        # compute the distance with multiple Lp distance metrics
        for i in range(len(self.p)):
            dist = torch.norm(batch_basis-x, p=self.p[i], dim=1, keepdim=True)
            x_d = x_d+dist/self.T[i] \
                if x_d is not None else \
                dist/self.T[i]
        
        return x_d

class CDR(nn.Module):
    """
        Centroid-based dimensionality reduction module
        It can be used to replace the fully-connected layer
    """

    def __init__(self, 
                    in_features=500, 
                    out_features=10,
                    normalized_radius=1.0, 
                    alpha=0.005, 
                    p=None, t=None):
        """
            Constructor for CDR
            Input
                in_features: input feature dimension
                out_features: output feature dimension
                normalized_radius: the radius used in normalization for initialization
                alpha: the influence control factor to balance dissimilarities of other labels
                p: a set of Lp distance metrics
                t: a set of temperatures w.r.t p
        """
        super(CDR, self).__init__()
        self.alpha = alpha
        self.t = t
        self.p = p
        self.centroids = nn.ModuleList([Centroid(in_features, normalized_radius=normalized_radius, p=self.p, t=self.t) for _ in
                         range(out_features)])

    def forward(self, x):
        x = [m(x) for m in self.centroids]
        x = torch.stack(x, dim=1).squeeze(2)
        if self.training:
            x = x - (torch.sum(x,dim=1,keepdim=True)-x)*self.alpha
        x = -x

        return x

    def getWeights(self):
        """
            A function to get all the centroids
        """
        w = None
        w_list = []
        for i, m in enumerate(self.centroids):
            w_list.append(m.basis.detach().unsqueeze(0))
        w = torch.cat(w_list, dim=0)

        return w