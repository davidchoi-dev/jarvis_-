#!/usr/bin/env python
# coding: utf-8

# In[402]:


# 자비스 모델 학습
# torchaudio 라이브러리 사용은 아래를 참조함
# https://tutorials.pytorch.kr/beginner/audio_preprocessing_tutorial.html
import os
import torch
import torchaudio
# MelSpectrogram은 이미지 형식의 데이터이기 때문에 CNN 을 사용하기 전에 이미지 전처리 해주는 라이브러리를 임포트
# 아직은 안쓸거다.
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 인공신경망 모델 사용
import torch.nn as nn
import torch.nn.functional as F
# 최적화
import torch.optim as optim
get_ipython().run_line_magic('matplotlib', 'inline')
class_dic = {}


# In[403]:


class Net(nn.Module):
    global class_dic
    def __init__(self):
        super(Net, self).__init__()
        # convolution
        self.conv1 = nn.Conv2d(3,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        # flully connected
        self.fc1 = nn.Linear(16*23*23,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,len(class_dic))
    def forward(self,x):
        # convlution 신경망 feed forward. max pooling 커널 크기는 2
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        # fully coneected
        x = x.view(-1,self.num_flat_features(x))
        #x = x.view(-1,16*23*23)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[405]:


# 데이터를 가져와 텐서로 바꿈. path는 './data' 형태로 넣어줌.
def getTrainData(path):
    file_paths = os.listdir(path)
    path = path+'/'
    datas = []
    class_dic = {} # 딕셔너리는 "화자명"=>번호(int) 쌍이다.
    classes = []
    labels = []
    for file_path in file_paths:
        waveform, sample_rate = torchaudio.load(path+file_path)
        data_MelSpec = torchaudio.transforms.MelSpectrogram()(waveform) # 학습에 데이터 특징점을 이용하기 위해 MelSpectrogram으로 변환
        data = data_MelSpec[0:1,:100,:100] # CNN에 사용하기 위해 100 * 100 로 잘라줌, 데이터 손실 발생하며 보안 방안이 필요하다.
        data = torch.cat(list(torch.split(data,1,dim=0))*3)
        name = file_path.split('_',maxsplit=-1)[0] # 발화자명
        datas.append(data) # 1*128*128 데이터 하나 추가
        if(name not in class_dic):
            class_dic[name] = len(class_dic) # 집합에 발화자명 추가. 총 클래스 수(이름 종류)를 알 수 있다.
        classes.append(name)
    print('발화자 수는 : %s 명 입니다.' %(len(class_dic)) )
    # 데이터 첫 번째부터 마지막까지
    for i,name  in enumerate(classes):
        idx = class_dic[name] # 딕셔너리에서의 idx
        labels.append(idx)
    print('레이블 크기는 : %s 입니다' %(len(labels) ))
    return datas, labels, class_dic # 데이터셋, 레이블, 클래스딕셔너리를 반환한다. 클래스 수는 softmax에서 이용한다.


# In[421]:


# 데이터를 가져와 텐서로 바꿈. path는 './data' 형태로 넣어줌.
def getTestData(path):
    global class_dic# 딕셔너리는 "화자명"=>번호(int) 쌍이다.
    file_paths = os.listdir(path)
    path = path+'/'
    datas = []
    classes = []
    labels = []
    for file_path in file_paths:
        waveform, sample_rate = torchaudio.load(path+file_path)
        data_MelSpec = torchaudio.transforms.MelSpectrogram()(waveform) # 학습에 데이터 특징점을 이용하기 위해 MelSpectrogram으로 변환
        data = data_MelSpec[0:1,:100,:100] # CNN에 사용하기 위해 100 * 100 로 잘라줌, 데이터 손실 발생하며 보안 방안이 필요하다.
        data = torch.cat(list(torch.split(data,1,dim=0))*3)
        name = file_path.split('_',maxsplit=-1)[0] # 발화자명
        datas.append(data) # 1*128*128 데이터 하나 추가
        classes.append(name)
    print('발화자 수는 : %s 명 입니다.' %(len(class_dic)) )
    # 데이터 첫 번째부터 마지막까지
    for i,name  in enumerate(classes):
        if class_dic.get(name):
            idx = class_dic[name]
        else:
            idx = -1
        labels.append(idx)
    print('레이블 크기는 : %s 입니다' %(len(labels) ))
    return datas, labels # 데이터셋, 레이블을 반환한다.


# In[414]:


# DataLoader를 위해 맞춤형 Dataset 클래스 생성.
class Jarvis_Data(Dataset):
    def __init__(self, x_data, y_data, transform=None):
        self.x_data = x_data # 이미 텐서 형태로 가져온 데이터 이기에 1, 128, 128 차원이다. permute 해줄 필요 없음
        self.y_data = y_data
        self.transform = transform
        self.len = len(y_data)

    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len

class DataTransform:
    def __call__(self, sample):
        inputs, labels = sample
        labels = torch.tensor(labels)
        transf = transforms.Compose([transforms.ToPILImage(),transforms.Resize(100),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
        final_output = transf(inputs)

        return final_output, labels


# In[407]:


# 데이터를 가져오고 데이터 로더를 초기화 한다.
datas, labels, class_dic = getTrainData('./data_train')
train_data = Jarvis_Data(datas,labels, transform=DataTransform())
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)


# In[408]:


# 네트워크 초기화 = 객체생성 및 loss function 정의, 최적화 정의
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(net)


# In[409]:


for epoch in range(1):
    # loss 값
    running_loss = 0.0

    for i, data in enumerate(train_loader, 0):
        # Feed Forward
        inputs, labels = data
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # Backward Propagation
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 20 == 19: # 총 65개의 미니배치 중 20 번 째일 때 loss 값을 출력
            print('[%d, %5d] loss: %.3f '%(epoch+1, i+1, running_loss/20))
            running_loss = 0.0
print('Finishied Training')


# In[410]:


# 학습시킨 모델을 저장하기
PATH = './jarvis_net.pth'
torch.save(net.state_dict(),PATH)


# In[412]:


# 모델 불러오기
net = Net()
net.load_state_dict(torch.load(PATH))


# In[423]:


# 데이터를 가져오고 데이터 로더를 초기화 한다.
datas, labels = getTestData('./data_test')
print(len(datas))
test_data = Jarvis_Data(datas,labels, transform=DataTransform())
test_loader = DataLoader(train_data, batch_size=1, shuffle=True)


# In[426]:


correct = 0
total = 0
with torch.no_grad(): # back prop을 위한 computational graph 생성하지 않아 비용 효율적인 테스팅이 가능하다
    for data in test_loader:
        datas, labels = data
        outputs = net(datas)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) # batch의 크기 만큼
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on Voices : %d %%' %( 100*correct/total ))


# In[ ]:
