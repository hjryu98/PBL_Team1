# PBL_Team1
2023 1학기 산업공학캡스톤 PBL Team1의 코드입니다.

----------------
### Abstract
This project was conducted in Hanyang University's industrial engineering capstone course. This project is an applicable study that improves the performance of voice conversion by adding speaker encoder of AUTOVC to the StarGAN-VC model. Please refer to the contents of StarGAN-VC and AUTOVC below. <br>
AUTOVC: https://arxiv.org/abs/1905.05879 <br>
StarGAN-VC: https://arxiv.org/abs/1806.02169 

----------------
### Dependencies
- Python 3.7
- Pytorch 1.13.1
- librosa 0.6.3
- pyworld 0.3.0
- tqdm
- tensorboardX

----------------
### Usage
1. Download Dataset. Applied VCTK Corpus
```
이용할 데이터셋을 다운로드 받습니다. 이용한 데이터셋은 VCTK Corpus.
```
<br>

2. Preprocess
```
python preprocess.py --sample_rate 16000 \
                    --origin_wavpath data/VCTK-Corpus/wav48 \
                    --target_wavpath data/VCTK-Corpus/wav16 \
                    --mc_dir_train data/mc/train \
                    --mc_dir_test data/mc/test
```


3. Pre-train
```
데이터에서 speaker embedding vector를 추출
```


4. move_npy
```
발화자 별로 구별 되어 있던 npy. 파일을 train이라는 상위 폴더로 옮겨줌
```

<br>

5. Train model
```
python main.py
```

<br>

6. Convert
```
위에서 학습 시킨 모델을 통해 기존의 발화자 정보를 convert (Ex. 262번 발화자의 발화 내용을 마치 272번 발화자가 이야기 한 것처럼 convert)
```

&nbsp;
