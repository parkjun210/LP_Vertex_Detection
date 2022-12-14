# SNU: LP Vertex Detection



# 실행 결과 예시 

![sample](https://user-images.githubusercontent.com/68048434/190587798-a7f5b6dc-3ff2-4ae0-8130-c260fc739c53.jpg)

Green Vertex    : GT

Red Vertex      : Model prediction


# Environments
```
conda create -n ENV_NAME python=3.9

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install opencv-python

pip install tqdm

pip install Shapely
```


# Directory 설명
    |── data                : Image dataset 폴더
        |── images
            |── test
            |── train
        |── labels
            |── test
            |── train
    |── experiments         : 학습한 모델이 저장되는 폴더
    |── utils
        ├──> data_loader.py : pytorch custom dataset
        ├──> eval.py        : IOU calculation
        ├──> helpers.py
    |── result              : visualize.py 실행 결과가 저장되는 폴더 
    |
    |── config.py           : 입력 argument를 관리하는 파일
    |── model.py            : Model architecture
    |── train.py            : image에서 Alined face image (112 x 112)를 추출하는 코드
    |── visualize.py        : LP Vertex Detection inference




# 코드 실행 가이드 라인


## pre-trained ckpt


아래 링크에서 미리 학습한 ckpt 폴더(exp_0)를 다운 받아 "LP_Vertex_Detection/experiments" 폴더에 배치

Ex. "LP_Vertex_Detection/experiments/exp_0/ckpt.pth"

https://drive.google.com/drive/folders/1i9s28H6lThreD8x99kiMD2KQrwLiHkYv?usp=sharing

## 코드 실행

  아래 명령어를 통해 실행한다. 
 
  python visualize.py 
     
    python visualize.py 
    
