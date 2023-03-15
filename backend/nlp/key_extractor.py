# KoBERT 모델 로딩 및 설정
import torch
from transformers import AutoTokenizer, AutoModel


def getKeywordFromSentence(input):
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    model = AutoModel.from_pretrained("skt/kobert-base-v1")

    # 예시 문장 생성
    sentence = input

    # 입력 데이터 전처리 및 추론
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[0]

    # 출력 확인
    print(last_hidden_states)

    # 각 토큰의 중요도를 계산합니다.
    importance = torch.sum(last_hidden_states, dim=0)

    # 중요도가 높은 순서대로 정렬하여 상위 5개의 단어를 추출합니다.
    num_keywords = 5
    top_indices = importance.argsort(descending=True)[:num_keywords]
    keywords = [tokenizer.convert_ids_to_tokens([top_indices[i]])[0] for i in range(num_keywords)]

    print(keywords)

    return keywords