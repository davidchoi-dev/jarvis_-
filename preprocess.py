# -*- coding: utf-8 -*-
"""
Created on Mon May 25 02:15:52 2020

@author: qeadz
"""

# 파일 열기 및 저장
import os
import scipy.io.wavfile

# 파일 Augmentation
import numpy as np
import librosa
aud_rate = 22050
# plot을 쥬피터노트북에서 볼 수 있다.
# %matplotlib inline 

# wav 파일만 가능. HTML에서 wav파일을 녹음해 보낸다.
def load_file(path):
    input_length = 30000
    data = librosa.core.load(path)[0]
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant" )
    return data
    
# Aug1. 백색소음 추가
def data_aug_wn(data, rate=1): # 백색소음 추가 데이터
    rate = 0.001 * rate
    wn = np.random.randn(len(data))
    data = data + rate*wn
    return data
    
# Aug2. 소리 늘리기, rate를 0.8부터 1.2까지 여러개 생성하자.
def data_aug_stretch(data, rate=0):
    rate = 1 + 0.025 * rate
    data = librosa.effects.time_stretch(data, rate)
    return data

# Aug3. 소리 이동시키기.(배열에서 데이터를 좌=>우로 이동시킴. 맨오른쪽 영역은 왼쪽으로 들어감)
# 100에서 2000까지 여러개 생성하자
def data_aug_shift(data, rate=1):
    rate = 100 * rate
    data = np.roll(data, rate)
    return data

# 한 번에 모든 데이터셋을 Augmentation 하고 저장하기.
# path 형식 : './data'
# 총 rate * 5 개 만큼의 데이터가 생긴다.
def aug_restore_All(path, rate=1): #rate 10을 권장
    file_paths = os.listdir(path)
    path = path+'/'
    for file_path in file_paths: # 이제....augment할 단위를 여러개 주자.
        data = load_file(path+file_path)
        for r in range(1, rate):
            data_wn = data_aug_wn(data, rate=r)
            data_st = data_aug_stretch(data, rate=r)
            data_st_sub = data_aug_stretch(data, rate=-r)
            data_sh = data_aug_shift(data, rate=r)
            data_sh_sub = data_aug_shift(data, rate=-r)
            f_name = file_path.split('.',maxsplit=-1)
            scipy.io.wavfile.write(path+f_name[0]+"_wn"+str(r)+".wav",aud_rate,data_wn)
            scipy.io.wavfile.write(path+f_name[0]+"_st"+str(r)+".wav",aud_rate,data_st)
            scipy.io.wavfile.write(path+f_name[0]+"_st_sub"+str(r)+".wav",aud_rate,data_st_sub)
            scipy.io.wavfile.write(path+f_name[0]+"_sh"+str(r)+".wav",aud_rate,data_sh)
            scipy.io.wavfile.write(path+f_name[0]+"_sh_sub"+str(r)+".wav",aud_rate,data_sh_sub)
        