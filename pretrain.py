import os
import pickle
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn


class D_VECTOR(nn.Module):
    """d vector speaker embedding."""

    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell,
                            num_layers=num_layers, batch_first=True)
        self.embedding = nn.Linear(dim_cell, dim_emb)

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:, -1, :])
        norm = embeds.norm(p=2, dim=-1, keepdim=True)
        embeds_normalized = embeds.div(norm)
        return embeds_normalized


C = D_VECTOR(dim_input=80, dim_cell=768, dim_emb=256).eval().cuda()
c_checkpoint = torch.load('3000000-BL.ckpt')
new_state_dict = OrderedDict()  # 순서까지 인식하는 Dictionary 입력 클래스
for key, val in c_checkpoint['model_b'].items():  # 앞에 module. 을 뺀 새 Dictionary
    new_key = key[7:]
    new_state_dict[new_key] = val
C.load_state_dict(new_state_dict)  # pre_trained 모델에서 파라미터 값 가져옴
num_uttrs = 10
len_crop = 128

# Directory containing mel-spectrograms
rootDir = './data/mc/train'
dirName, subdirList, _ = next(os.walk(rootDir))

speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName, speaker)))

    # make speaker embedding
    assert len(fileList) >= num_uttrs
    idx_uttrs = np.random.choice(len(fileList), size=num_uttrs, replace=False)
    embs = []
    for i in range(num_uttrs):
        tmp = np.load(os.path.join(dirName, speaker, fileList[idx_uttrs[i]]))
        candidates = np.delete(np.arange(len(fileList)), idx_uttrs)
        # choose another utterance if the current one is too short
        while tmp.shape[0] < len_crop:
            idx_alt = np.random.choice(candidates)
            tmp = np.load(os.path.join(dirName, speaker, fileList[idx_alt]))
            candidates = np.delete(candidates, np.argwhere(candidates == idx_alt))
            # 모두 128보다 길어서 실행되지 않음
        left = np.random.randint(0, tmp.shape[0] - len_crop)
        melsp = torch.from_numpy(
            tmp[np.newaxis, left:left + len_crop, :]).cuda()  # 기존 (x,80)의 음성데이터를 torch size가 [1,128,80]인 고정 melsp로 변환
        emb = C(melsp.float())  # utterence 별로 torch size가 [1,256]인 고정 emb 생성
        embs.append(emb.detach().squeeze().cpu().numpy())
    utterances.append(np.mean(embs, axis=0))
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker, fileName))
    speakers.append(utterances)

#with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
#   pickle.dump(speakers, handle)


