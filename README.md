# SNU: LP Vertex Detection



# 실행 결과 예시 

![inf_result](https://user-images.githubusercontent.com/68048434/234839227-a1ab0599-532f-497d-badd-bc8ecd4f69be.jpg)

Red Vertex      : Model prediction


# Environments
```
git clone -b inference --single-branch https://github.com/parkjun210/LP_Vertex_Detection.git

conda create -n ENV_NAME python=3.8

conda activate ENV_NAME

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

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
    |── inference.py        : LP Vertex Detection inference




# 코드 실행 가이드 라인


## pre-trained ckpt


아래 링크에서 미리 학습한 ckpt.pth 파일을 다운 받아 "LP_Vertex_Detection/experiments" 폴더에 배치

Ex. "LP_Vertex_Detection/experiments/ckpt.pth"

https://drive.google.com/drive/folders/1i9s28H6lThreD8x99kiMD2KQrwLiHkYv?usp=sharing

## 코드 실행

  inference.py의 Argument 부분을 필요에 맞게 변경한다.

    GPU_NUM:        사용할 GPU number
    WEIGHT_PATH:    ckpt 파일 경로
    INFERENCE_DIR:  inferece할 폴더 경로
    SAVE_DIR:       inference 결과가 담길 폴더 경로
    SAVE_IMG_FLAG:  Inference result image 저장 여부

  아래 명령어를 통해 실행한다. 
     
    python inference.py 
