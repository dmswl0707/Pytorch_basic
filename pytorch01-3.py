##하이퍼파라미터 설정하기

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms #데이터를 모델에 전달(주로 배치를 이용)
from torch.utils.data import DataLoader

batch_size=256
learning_rate=0.0002
num_epoch=10

mnist_train=dset.MNIST("./", train=True, transform=transforms.ToTensor(), #이미지데이터를 파이토치 텐서
                       target_transform=None, download=True)
mnist_test=dset.MNIST("./",train=False,transform=transforms.ToTensor(),
                      target_transform=None, download=True)

train_loader=torch.utils.data.DataLoader(mnist_train,batch_size=batch_size,
                                         suffle=True, num_workers=2, drop_last=True)
test_loader=torch.utils.data.DataLoader(mnist_test, batch_size=batch_size,
                                        suffle=False, num_workers=2, drop_last=True)



