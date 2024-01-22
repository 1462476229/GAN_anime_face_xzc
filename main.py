import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import loaddata
import random

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        n_generator_feature = 128
        self.main = nn.Sequential(      
            nn.ConvTranspose2d(50, n_generator_feature * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_generator_feature * 8),
            nn.ReLU(True),      
            nn.ConvTranspose2d(n_generator_feature * 8, n_generator_feature * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_generator_feature * 4),
            nn.ReLU(True),     
            nn.ConvTranspose2d(n_generator_feature * 4, n_generator_feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_generator_feature * 2),
            nn.ReLU(True), 
            nn.ConvTranspose2d(n_generator_feature * 2, n_generator_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_generator_feature),
            nn.ReLU(True),      
            nn.ConvTranspose2d(n_generator_feature, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()       
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        n_discriminator_feature = 128
        self.main = nn.Sequential(
            nn.Conv2d(3, n_discriminator_feature, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),       
            nn.Conv2d(n_discriminator_feature, n_discriminator_feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),       
            nn.Conv2d(n_discriminator_feature * 2, n_discriminator_feature * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 4),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(n_discriminator_feature * 4, n_discriminator_feature * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 8),
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(n_discriminator_feature * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()      
        )

    def forward(self, input):
        return self.main(input).view(-1)
    
def train(D, G, data, epochs:int = 1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("是否链接上了显卡驱动: ", device)
    D = D.to(device)
    G = G.to(device)
    # G_opt = torch.optim.RMSprop(G.parameters(),lr=0.0005)
    # D_opt = torch.optim.RMSprop(D.parameters(),lr=0.0005)
    G_opt = torch.optim.Adam(G.parameters(),lr= 0.0002, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(D.parameters(),lr= 0.0002, betas=(0.5, 0.999))
    loss_function_MSE = nn.MSELoss()
    loss_function_BCE = nn.BCELoss()
    # loss_function_2 = nn.L1Loss()
    # L2范数距离来惩罚生成器。
    loss_function_L1 = nn.L1Loss()
    total_data = len(data)
    for epoch in range(epochs):
        for i, images in enumerate(data):
            # 把数据放入 device(显卡) 中，并预处理一下要用的数据
            images = images.to(device)
            batch_size = images.size(0)
            latent = torch.randn(batch_size, 50, 1, 1).to(device)
            # z = torch.randn_like(images).to(device)
            # eps = random.uniform(0.0, 1.0)
            # random_images = (1 - eps) * images + eps * z
            # images_z = images + z * 0.1 # 加入高斯噪声相当于是增加了
            # print("批大小: ",batch_size)
            # 生成虚假和真实标签，为了求 loss 而存在的
            real_label = torch.ones(batch_size).to(device)
            # fake_label = torch.zeros(batch_size, 1).to(device)
            # 先判断 G 对真实图片的 loss 损失
            if i % 1 == 0:
                
                d_output = D(images)
                # print(f"d_output {d_output.shape}")
                # print("d_output: ",d_output.shape, real_label.shape)
                d_loss_real = loss_function_BCE(d_output, real_label * 0.95) # 小技巧，不过于自信不设置满了
                # d_loss_real = d_output.mean()
                # 技巧 Loss 不取 Log 直接取
                
                # 再判断 G 对虚假图片的 loss 损失
                fake_images = G(latent)      
                d_output = D(fake_images.detach())
                d_loss_fake = loss_function_BCE(d_output, real_label * 0.01)
                # d_loss_fake = d_output.mean()
                
                # D_opt.zero_grad() # 梯度损失惩罚
                # d_output = D(random_images)
                # d_output.mean().backward()
                # grad = torch.zeros(1).to(device)
                # param_sum = 0
                # for param in D.parameters(): # 计算梯度和, 和网络参数的总个数
                #     grad += torch.pow((torch.sqrt(torch.pow(param.grad, 2)) - 0.2), 2).sum()
                #     param_sum += param.grad.numel()
                # print(grad, param_sum)
                # grad_loss = grad / param_sum
                # print(grad_loss)
                # d_loss = (10 * grad_loss + d_loss_fake - d_loss_real) # 两个 loss 损失相加取一个平均,  而且我这里抑制 判别器 过早收敛
                # d_loss = d_loss_fake - d_loss_real
                
                d_loss = d_loss_real + d_loss_fake
                D_opt.zero_grad()
                d_loss.backward()
                # 梯度裁剪
                # torch.nn.utils.clip_grad_norm_(parameters=D.parameters(), max_norm=10, norm_type=2)
                D_opt.step()
                # for p in D.parameters(): 直接裁剪参数
                #     p.data.clamp_(-0.01, 0.01)
                # 自此对神经网络 D 的训练完成
                
            if i % 5 == 0:  
                fake_image = G(latent)
                g_output = D(fake_image)
                # g_loss = 3 * loss_function_MSE(g_output.squeeze([2,3]), real_label) + loss_function_MSE(fake_image, images_L)
                # nn.Loss 不是一个函数, 而是一个类得提前定义
                # g_loss = -g_output.mean() #+ 0.1 * loss_function_MSE(fake_image, images)
                g_loss = loss_function_BCE(g_output, real_label)
                
                G_opt.zero_grad() # 这里的清零很重要
                g_loss.backward()
                G_opt.step()
                # 自此对神经网络 G 的训练完成
            if i % 100 == 0:
                # if epoch % 2 == 0 and i == 100:   
                #     image_L = torch.randn_like(images[0:1, 0:1, :, :]).to(device)
                #     output = G(image_L)
                #     image = torch.cat([output[0,0:1,:,:].to("cpu").detach(), torch.zeros(2,128,128)], dim=0)
                #     LoadData.ShowOneTensor(image,"Process", mean = mean, std = std, isLab = True)
                    
                print(f"训练到 {epoch}/{epochs} , {i}/{total_data} , 误差D {d_loss.sum()} , 误差G {g_loss.sum()}")  
                with torch.no_grad():
                    latent = torch.randn(batch_size, 50, 1, 1).to(device)
                    output = G(latent)
                    fake_image_p = D(output.detach())
                    real_image_p = D(images[0:1,:,:,:])
                print(f"单个照片的准确度关于 D 的::  fake:{int(fake_image_p.sum() * 100)} %  real:{int(real_image_p.sum() * 100)} %")
                # print(f"单个照片的准确度关于 D 的::  fake:{int((1-fake_image_p.sum()) * 100)} %  real:{int((1-real_image_p.sum()) * 100)} %") # WGAN  
        if epoch != 0 and epoch % 500 == 0:
                
            torch.save(D, f"./Net/Final_D_{epoch}.pth") # 把网络 D 存进去
            torch.save(G, f"./Net/Final_G_{epoch}.pth") # 把网络 G 存进去
        
        
if __name__ == "__main__":
    data , mean, std = loaddata.creatDataset("./data",batchsize=128)
    # data , mean, std = loaddata.creatDataset("F:\\DataSet\\dongman-512\\anime_face",batchsize=64)
    print(mean, std)
    G = Generator()
    D = Discriminator()
    #G = torch.load("./Net/final_G_850.pth")
    #D = torch.load("./Net/final_D_850.pth")
    train(D, G, data, 5000)