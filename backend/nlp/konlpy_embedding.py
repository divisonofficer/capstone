from konlpy.tag import Okt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Specify your file path


file_path = '../crawling/merged.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
df = df.dropna(subset=['Content'])


print("Loading KoBERT")


okt = Okt()
# Tokenizing
tqdm.pandas(desc="Tokenizing")
df['Morphs'] = df['Content'].apply(lambda x: Okt.morphs(x))
# 형태소들의 리스트를 모두 합쳐서 고유한 형태소들의 집합을 만듦
all_morphs = set()
for morphs in df['Morphs']:
    all_morphs.update(morphs)

# 각 형태소에 고유한 ID 할당
token_to_id = {morph: idx for idx, morph in enumerate(all_morphs)}


# 정수 벡터 변환 및 패딩을 위한 함수 정의
def convert_to_int_vector(morphs, token_to_id, max_length=500):
    # 형태소를 정수로 변환
    int_vector = [token_to_id.get(morph, 0) for morph in morphs]

    # 벡터 길이 조정
    if len(int_vector) < max_length:
        # 짧은 경우 0으로 패딩 추가
        int_vector += [0] * (max_length - len(int_vector))
    else:
        # 긴 경우 초과 부분 제거
        int_vector = int_vector[:max_length]

    return int_vector

# 'Morphs' 열에 정수 벡터 변환 및 패딩 적용
tqdm.pandas(desc="Int-Vectorizing")
df['Int_Vectors'] = df['Morphs'].apply(lambda x: convert_to_int_vector(x, token_to_id))


