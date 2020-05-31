# jarvis_-
화자인식 머신러닝 모델입니다.

0. 녹음 파일을 준비해 주세요. (자비스~ 라고 불러주심 됩니다. 녹음 길이는 2~2.5초 사이가 적당합니다.)
루트 폴더에 data_test , data_train 폴더를 생성해 주세요.

scipy.io.wavfile ( wav 파일 입출력 라이브러리 )
librosa ( 음성 파일 data augmentation )
numpy
torch
torchvision ( 머신러닝 시에 CNN 을 사용할 것입니다. )
torchaudio ( wav 파일 로드 시점에서 바로 tensor 로 변환시켜 줍니다. 또한, 음성 신호 추출에 유리한 포맷으로의 변환하는 기능이 많습니다. 여기선 MelSpectrogram을 사용했습니다. )

라이브러리들을 미리 다운받아 주세요.

c.f.) torchaudio 라이브러리 윈도우 설치 방법입니다.
1) pip install pipwin
2) pipwin install pyaudio

1. 녹음 파일은 wav 으로 변환시켜 data_train 폴더에 저장해 주세요. 
(c.f. dhsimpson_(1) dhsimpson_(2) .... 이런 식으로 이름 뒤에 언더바를 넣어주셔야 해요. 클래스 분류 시에 첫 번째 언더바 앞부분만 이름으로 사용합니다. )
2. 친구에게 자비스~ 라고 음성 녹음 몇 개만 해달라고 부탁하세요. (친구가 없는 저는 스스로 목소리변조를 했습니다.)
(c.f. 친구의 목소리도 마찬가지로 위와 같이 저장해 줍니다. 이름은 뭐로 해도 상관없어요. train 할 때 있던 이름이 아니면 전부 딕셔너리에서 None 으로 처리시킬 겁니다.
3. 2번의 파일과 1번의 파일 몇 개를 추려 data_test 폴더에 저장해 주세요.
4. makeTrainSet.py를 실행시키면 preprocess.py 의 함수를 실행해 data_train 폴더의 wav 파일들을 변조(augmentation) 시켜줍니다.(데이터 셋을 늘려줍니다.)
5. trainModel.py 를 실행시키면 data_train 폴더의 wav 파일들을 이용해 학습한 뒤 data_test 폴더의 wav 파일들을 이용해 테스트 합니다.


trainModel.py에 대해

1. trainModel 이 실행되고 data_train 폴더의 파일들을 로드하면 MelSpectogram으로 변환시킨 뒤 [3,100,100] 크기의 텐서로 만들어 줍니다. 
( 음성파일 크기가 제각각이라 앞의 100 만큼 씩만 슬라이싱해 가져옵니다. )
( 원래는 MelSpectogram의 차원은 [1,100,100] tensor 이지만 파이토치의 CNN에서 사용하려면 input이 3차원이어야 한다더라구요. )
2. 신경망 구조는 CNN 을 사용했으며, 출력단은 softmax 입니다. (loss function은 cross-entropy)
3. 학습시킬 때 이름들은 딕셔너리(class_dic)에 넣어 softmax에 이용할 수 있도록 해줬습니다.
4. 테스트할 때 class_dic에 없는 키 값(이름) 은 None key로 생각해서 분류합니다.

프론트 엔드까지 만들게 되면 다시 돌아올게요.
