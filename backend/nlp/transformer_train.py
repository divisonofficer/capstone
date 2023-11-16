from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

device = torch.device("cuda")

def tokenize_data(train_x, train_y):
    input_ids = []
    attention_masks = []
    labels = []

    for x, y in zip(train_x, train_y):
        encoded_dict = tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            max_length=256,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        encoded_label = tokenizer.encode_plus(
            y,
            add_special_tokens=True,
            max_length=256,  # 레이블의 최대 길이를 설정합니다.
            pad_to_max_length=True,
            return_tensors='pt',
        )
        labels.append(encoded_label['input_ids'][0])

    input_ids = torch.stack(input_ids, dim=0).to(device)

    #[2257, 1, 256] -> [2257, 256]
    input_ids = input_ids.squeeze()

    attention_masks = torch.stack(attention_masks, dim=0).to(device)
    attention_masks = attention_masks.squeeze()
    labels = torch.stack(labels, dim=0).to(device)
    
    print(input_ids.shape, attention_masks.shape, labels.shape)

    return input_ids, attention_masks, labels

from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(tokenizer.vocab),
)
model.to(device)


print(model.config)

import pandas as pd

df = pd.read_csv('merged.csv')

input_ids, attention_masks, labels = tokenize_data(df['Title'], df['Content'])





from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW

# 훈련 데이터 로더 생성
batch_size = 32
epochs = 10
train_data = TensorDataset(input_ids, attention_masks, labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# 옵티마이저 설정
optimizer = AdamW(model.parameters(), lr=2e-5)

# 훈련 루프
for epoch in range(epochs):
    model.train()
    total_loss = 0
    print(f'Epoch {epoch + 1}/{epochs}')
    for batch in train_dataloader:
        b_input_ids, b_attention_masks, b_labels = batch
        
        print(b_input_ids.shape, b_attention_masks.shape, b_labels.shape)

        model.zero_grad()
        outputs = model(b_input_ids, attention_mask=b_attention_masks, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Average training loss: {total_loss / len(train_dataloader)}')



def evaluate(sentence):
    # 토큰화
    inputs = tokenizer(
        sentence,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=64,
        add_special_tokens=True
    )
    
    # 모델을 통과
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 결과를 디코딩
    # 출력이 로짓이므로 argmax를 사용하여 가장 높은 값의 인덱스를 찾습니다.
    predicted_index = torch.argmax(outputs.logits, dim=-1).item()
    predicted_label = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    
    return predicted_label

# 예제
sentence = "안녕하세요, 반갑습니다."
#result = evaluate(sentence)
#print(result)