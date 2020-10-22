##기울기 최적화하기

#torch.tensor 함수는 data, dtype, device, requires_grad을 인수로 받음


import torch

#requires_grad는 기울기 저장할지 여부(기울기 계산)
x_tensor=torch.tensor(data=[2.0,3.0], requires_grad=True)
y=x**2
z=2y+3 #결괏

target=torch.tensor([3.0],[4.0])값 #목푯값

loss=torch.sum(torch.abs(z-target)) # 오차값 합
loss.backward() #그래프 따라가면 기울기 계산

print(x.grad, y.grad, z.grad) # x none none


