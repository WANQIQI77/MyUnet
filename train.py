import numpy as np

from model.unet_model import UNet
from utils.DataSet import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def Train_Net(net, device, data_path, epochs=40, batch_size=1, lr=0.000001):
    """
    The founction of the Net training
    :param net: The Net classification
    :param device: CPU or GPU
    :param data_path: The path of the data
    :param epochs: epochs number
    :param batch_size: batch_size
    :param lr: Index weighted average
    :return: None
    """
    isbi_dataset = ISBI_Loader(data_path)
    per_epoch_num = len(isbi_dataset) / batch_size
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    # 定义RMSprop算法

    # 1.换学习器？？
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    
    
    # 定义Loss算法
    # criterion = nn.BCEWithLogitsLoss()#二分类问题、
    criterion = nn.CrossEntropyLoss(reduction='sum')
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')

    # 添加可视化损失变化的部分
    losses = []
    epochs_list = []

    with tqdm(total=epochs*per_epoch_num) as pbar:
        for epoch in range(epochs):
            net.train()
            # According to batch_ Size Start training
            for ori_image, ori_label in train_loader:
                optimizer.zero_grad()
                ori_image = ori_image.to(device=device, dtype=torch.float32)
                ori_label = ori_label.to(device=device, dtype=torch.float32)
                # Use network parameters to output prediction results
                pred = net(ori_image)

                # 在计算损失之前，将目标张量调整为3D
                # ori_label = ori_label.squeeze(0)  # 去掉 batch 维度

                loss = criterion(pred, ori_label)
                print('{}/{}:Loss/train'.format(epoch + 1, epochs), loss.item())
                # Save the network parameter with the lowest loss value
                if loss < best_loss:
                    best_loss = loss
                    torch.save(net.state_dict(), 'epoch={}_model_UNet_8_1e-6.pth'.format(epochs))
                # Update parameters
                loss.backward()
                optimizer.step()
                pbar.update(1)

                # 记录损失值和epoch，用于可视化
                losses.append(loss.item())
                epochs_list.append(epoch * per_epoch_num + len(losses))
    # 可视化损失变化
    plt.plot(epochs_list, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('b=8,lr=1e-6.png')
    # plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_net = UNet(n_channels=3, n_classes=7)  # todo edit input_channels n_classes
    training_net.to(device=device)
    data_path = "TrainTarget/"

    # 2.修改学习率 batch_size对比
    Train_Net(training_net, device, data_path, epochs=100, batch_size=8, lr=1e-6)

    # 3.大模型 paddle-x 还有其他