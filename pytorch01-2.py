##선형회귀분석 모델

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

num_data=1000
num_epoch=500

x=init.uniform_(torch.Tensor(num_data,1),-10,10)
#가우시안 노이즈_표준편=1, (num_data,1)형태 초기화
noise=init.normal(torch.FloatTensor(num_data,1),std=1)
y=2x+3 #-17과 23사이의 분
y_noise=2*(x+noise)+3

model=nn.Linear(1,1)
loss_func=nn.L1Loss()
optimizer=optim.SGD(model.parameters(), lr=0.01) #model.parameters() w와 b를 전달 #torch.optim

label=y_noise
for i in range(num_epoch):
    optimizer.zero_grad() #기울기를 최소화하는 함
    output=model(x)

    loss=loss_func(output,label)
    loss.backward()
    optimizer.step() #각 파라미터 최적화하여 업데이

    if i%10==0:
        print(loss.data)

param_List=list(model.parameters())
print(param_List[0].item(), param_List[1].item())






