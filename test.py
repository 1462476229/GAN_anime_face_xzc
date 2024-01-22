import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import loaddata
from main import Generator ,Discriminator

def testG(G, mean, std, epochs:int = 1):
    """
    给两个网络 D, G 对这两个网络进行训练
    epochs 的默认大小为 1
    """
    with torch.no_grad():
        G = G.to("cpu")
        print(f"mean:{mean} std:{std}")
        for epoch in range(epochs):
            latent = torch.randn(1, 50, 1, 1)
            output = G(latent)
            loaddata.ShowOneTensor(output[0,...],"Process", mean = mean, std = std, isLab = 0, savePath=f"./Gen/{epoch}.jpg")
            
if __name__ == "__main__":
    G = torch.load("./Net/Final_G_1500.pth")
    # D = torch.load("D_anime_face_12.pth")
    mean = [0.564923882484436, 0.5881562829017639, 0.6970292925834656] 
    std = [0.2383042722940445, 0.25863999128341675, 0.2565942108631134]
    # data, _, __ = loaddata.creatDataset("./data-test", batchsize=1,istrain=0,mean=mean,std=std)
    testG(G, mean, std, 100)
