##CNN 모델

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__() #CNN 클래스의 부모 클래스인 nn.module을 초기
        self.layer=nn.Sequential(
            nn.Conv2d(1,16,5), #in_channels=1, out_channels=16, kernel_size=5
            nn.ReLU(), #.......................1
            nn.Conv2d(16,32,5),
            nn.ReLU(), #.......................2
            nn.MaxPool2d(2,2), #...............3
            nn.Conv2d(32,64,5),
            nn.ReLU(), #.......................4
            nn.MaxPool2d(2,2) #................5
        )
        self.fc_layer=nn.Sequential(
            nn.Linear(64*3*3,100),
            nn.ReLU(),
            nn.Linear(100,10) #형태 바꿔주기
        )

    ###nn.Conv2d - in_channels, out_channels, kernel_size, stride, padding
    ###nn.MaxPool2d - kernel_size, stride, padding

    # 파이토치의 첫걸음 3차원의 이해를 참조할 것
    # 합성곱의 입력은 [batch_size, in_channels, 가로, 세로]
    # 합성곱의 결과값은 [batch_size, out_channels, 가로, 세로]
    # 첫 입력값은 [batch_size, 1, 28, 28] , stride=1, padding=0으로 고
    # nn.Conv2d(1,16,5) 연산 결과 = [batch_size, 16, 24, 24]정
    # 100개의 카테고리에서 10개의 카테고리로 뉴런을 줄여나감

    # 1 [batch_size, 16, 24, 24]
    # 2 [batch_size, 32, 20, 20]
    # 3 [batch_size, 32, 10, 10]
    # 4 [batch_size, 64, 6, 6]
    # 5 [batch_size, 64, 3, 3]

    def forward(self,x): #순차적으로 실행하여 결괏값만 리턴
        out=self.layer(x)
        out=out.view(batch_size, -1) #............1
        out=self.fc_layer(out)
        return out

    # 결과값 [batch_size, 64, 3, 3]을 view 함수로 받음
    # view 함수에 인수로 목표로 하는 [batch_size, -1]를 전달, -1은 -1은 버리라는 의미, tensor.view(2, -1)
    # 첫 입력값은 [4, 16]
    # 1 [2, 32]









