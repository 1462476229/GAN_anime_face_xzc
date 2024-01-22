import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import cv2

def ShowOnePicture(x, savePath = None, grey = False): # 给一个地址，取查看他的 Lab 通道的图像
    """    
    用 cv2 库来显示一张图片, 用 q 退出. 
    Args:
        x:  图片的地址
            torch.Tensor 张量(BGR) (X,Y,C) or (C,X,Y) 范围在 0~255
            图片的 numpy 数组(BGR) (X,Y,C) or (C,X,Y)
        savePath (_type_, optional): 保存地址. Defaults to None.
        grey (bool, optional): 是否展示灰度图像. Defaults to False.
    """
    print(x.shape, type(x)) # 看一下内容
    if type(x) == str:
        img = cv2.imread(x) # 读入一个
        cv2.imshow("Image", img)
        Lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB) # 把 BGR 通道转化为 Lab 通道
        if grey == True:
            L, a, b = cv2.split(Lab) # 分一下通道输出            
            cv2.imshow("Image-L", L) # 输出 L 通道的值, 相当于灰度图
            
    elif type(x) == torch.Tensor: # torch 张量
        print(x)
        x = x.to("cpu").numpy() # 还是选择优先转化成 ndarray 再处理的
        if x.shape[2] != 3: # 可能会出现维度的问题, 这里直接优先处理了
            x = np.transpose(x, [1,2,0])
        cv2.imshow("Image", x)
        if grey == True:
            L, a, b = cv2.split(x)  
            cv2.imshow("Image-L", L)  
           
    elif type(x) == np.ndarray: # numpy 数组
        if x.shape[2] != 3:
            x = np.transpose(x, [1,2,0])
        cv2.imshow("Image", x)
        if grey == True:
            L, a, b = cv2.split(x)  
            cv2.imshow("Image-L", L)  
    else:
        return
    
    if savePath != None: # 保存路径
        cv2.imwrite(savePath, img=x)       
        return
    while True:    
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            cv2.destroyAllWindows()
            break # 输出输出
 
def ShowOneTensor(x : torch.Tensor, TensorType = "Normal",mean = None,std = None, isLab = False, savePath = None):
    """    
    还是显示一张图片, 只接受 torch.Tensor 的类型
    Args:
        x (torch.Tensor): 张量
        TensorType (_type_, optional): 
            考虑到不同 Tensor 的类型的:
                "Normal" 正常的 0 到 255 的图像
                "Process" 处理后在 0 到 1 的图像
        mean (_type_, optional): 数据集的三通道的平均值. Defaults to None.
        std (_type_, optional): 数据集的三通道的方差. Defaults to None.
        std (bool, optional): 图片是不是 Lab 通道的. Defaults to False.
    """
    if TensorType == "Normal":
        ShowOnePicture(x, savePath=savePath) # 直接调用
        
    elif TensorType == "Process":
        x = x.to("cpu").numpy()
        if x.shape[2] != 3: # 处理一下有可能是 [C,X,Y] 
            x = np.transpose(x, [1,2,0]) # 得变化为 [X,Y,C] 再操作
            
        if mean == None and std == None:
            x = (x + 1) / 2 * 255
            x = (np.uint8(x)) # 反归一化, 且这里的归一化用的 mean = [0.5 0.5 0.5], std = [0.5 0.5 0.5] 
        else:
            for d in range(3): # 反归一化
                x[:,:,d] *= std[d]
                x[:,:,d] += mean[d]
            x = x * 255
            x = (np.uint8(x))
            
        if isLab == True:
            x = cv2.cvtColor(x, cv2.COLOR_LAB2BGR) # 进来的是 Lab 的，得换成 BGR 通道才可以   
        ShowOnePicture(x, savePath=savePath)
    else:
        return
    
def ProcessData(para, kind = 'BGR') -> tuple[list, list, list]:
    """
    预处理数据, 包括图片弄成相同大小, 和计算出 平均值 和 方差
    Args:
        para (_type_): 
            数据集的路径: 只会读取文件夹里面的 .jpg 和 .png 图片
            图片集合的列表
        kind :
            LAB 或者 BGR
    Returns:
        list (_type_): 
                    第一个是数据的列表(长度为 num 的列表, 每一个元素是 [X,Y,C] 的 np.ndarray)  
                    第二个是 mean(平均值) 的列表
                    第三个是 std(标准差) 的列表
    """
    picList = []    
    if type(para) == str:
        root_path = para
        files = os.listdir(root_path) # 遍历所有文件，读取图片
        pictureId = 1
        for file in files:
            path = root_path + '/' + file
            fileSuffix = file.split('.')[-1] # 得到文件后缀
            if fileSuffix == "jpg" or fileSuffix == 'png':
                img = cv2.imread(path) 
                img = cv2.resize(img, (96, 96)) # 大小变为 (128 x 128)
                if kind == "Lab" or kind == "LAB":
                    picture = cv2.cvtColor(img, cv2.COLOR_BGR2Lab) # 进来是 BGR ，得换为 Lab 通道更好
                if kind == "BGR":
                    picture = img
                print(f"成功加载了第 {pictureId} 张图片")
                pictureId += 1
                picList.append(picture)
    elif type(para) == list:
        picList = para
    
    print("compute mean and variance for data.")
    print(f"数据的长度为 {len(picList)}")
    clac_picList = [] # 用一个临时的数组来存储
    
    ## Waring 我认为这里的计算还是稍微有点问题的 std 的计算不应该直接是所有的 std 求平均
    ## 考虑到误差不是很大就不管了
    for x in picList:
        clac_picList.append(transforms.ToTensor()(x))
    train_loader = torch.utils.data.DataLoader(
        clac_picList, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X in train_loader:
        for d in range(3): # 对三个通道处理
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(picList))
    std.div_(len(picList))
    
    return picList, mean.tolist(), std.tolist()
                
class Mydataset(Dataset):
    def __init__(self, data, mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5]):
        self.data = data # 数据这里严格要求 X,Y 大小相同, 而且是 Lab 的 [C,X,Y] 数据范围在 0~255 的矩阵
        transform = transforms.Compose([
            # Converts numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
            transforms.ToTensor(), 
            transforms.Normalize(mean=mean, std=std),  # Lab 图像三种颜色
            transforms.RandomHorizontalFlip(p = 0.5),
            # transforms.RandomVerticalFlip(p = 0.5)
        ]) # 自己定义的归一化，把大小都弄成相同的 512 x 512
        self.transform = transform # 转变方法

    def __getitem__(self, item): 
        img = self.data[item]
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)            
    
def creatDataset(path: str,batchsize = 1, istrain = True, mean = None, std = None) -> tuple[DataLoader,list,list]:
    """
    给出数据集地址, 返回可迭代的 DateLoader 和 mean 和 std, 也要给出 batchsize 的大小
    Args:
        path (str): 文件夹路径
        batchsize : 批大小
        istrain : 是否是训练数据, 如果是预测请带上 mean 和 std (训练数据集的)
    Returns:
        tuple[DataLoader,list,list]: 
           第一个可迭代的 DataLoader
           第二个是数据集归一化之后的 mean 
           第三个是数据集归一化之后的 std
    """
    if istrain:
        data, mean, std = ProcessData(path, kind='BGR') # 预处理图片
        data = Mydataset(data, mean=mean, std=std) # 得到自定义数据集
        train_data = DataLoader(data, batch_size=batchsize, shuffle=True, num_workers=0) # bathsize 和 numworkers 简单取决于电脑性能
        # for i, X in enumerate(train_data):# 以下是测试代码
        #     print(X,X.shape)
        #     ShowOneTensor(X[0,:,:,:],"Process", mean, std, isLab=True)
        #     print(mean, std)
        return train_data , mean , std
    else :
        data, _, __ = ProcessData(path, kind='BGR') # 预处理图片
        data = Mydataset(data, mean=mean, std=std) # 得到自定义数据集
        train_data = DataLoader(data, batch_size=batchsize, shuffle=True, num_workers=0) # bathsize 和 numworkers 简单取决于电脑性能
        return train_data , mean , std        
              
    


if __name__ == "__main__":
    # 测试 ShowOnePicture 函数代码
    # ShowOnePicture("./test/002.png")
    # ShowOnePicture("./test/002.png", "./test/003.png")    
    
    # img = cv2.imread("./test/001.jpg")
    # ShowOnePicture(img, grey=1)
    
    # img_torch = cv2.imread("./test/002.png")
    # img_torch = torch.tensor(img_torch.transpose([2,0,1])).to("cuda")
    # ShowOnePicture(img_torch, grey=1)

    # 测试 ProcessData 函数代码
    a,b,c=creatDataset("./test",batchsize=5,istrain=1)
    print(a,b,c)    