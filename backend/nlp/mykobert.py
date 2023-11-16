import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import sys
import konlpy
from torch.optim.lr_scheduler import ReduceLROnPlateau


HIDDEN_SIZE = 256

device = torch.device("cuda")

MODEL_FILENAME = "mymodel.pt"

def kobert_model():
    modelname = "klue/bert-base" #"kykim/bert-kor-base" #'monologg/kobert'
    tokenizer = BertTokenizer.from_pretrained(modelname)    
    model = BertModel.from_pretrained(modelname)
    model.to(device)
    model.eval()

    return model, tokenizer


bert_model = None
bert_tokenizer = None

def get_bert_embedding(text):
    # if text is short, repeat it until it reaches 512 tokens
    if len(text) < 512:
        text = (text + " ") * (512 // len(text) + 1)
    global bert_model, bert_tokenizer
    inputs = bert_tokenizer.encode_plus(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    inputs.to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs['last_hidden_state'][:,0,:]

import torch.nn.functional as F
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class KoBertEmbedding(nn.Module):
    def __init__(self):
        super(KoBertEmbedding, self).__init__()
        # 인코더 부분
        self.encoder = nn.Sequential(
            nn.Linear(768, 512),
            Mish(),
            nn.Linear(512, 256),
            Mish(),
            nn.Linear(256, HIDDEN_SIZE)  # 잠재 벡터 크기는 128
        )
        # 디코더 부분
        self.decoder = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 256),
            Mish(),
            nn.Linear(256, 512),
            Mish(),
            nn.Linear(512, 768)  # 최종 출력은 원래 입력 크기와 동일
        )

    def forward(self, x):
        x = self.encoder(x)  # 인코딩
        x = self.decoder(x)  # 디코딩
        return x

    def get_latent_vector(self, x):
        return self.encoder(x)  # 잠재 벡터를 얻는 함수
    

    

num_epochs = 100
model = None

def train_model(train_x, train_y):
    global model
    model = KoBertEmbedding().to(device)
    # 모델 인스턴스 생성 및 GPU로 이동


    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CosineSimilarity(dim=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.04)
    #scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)


    #batch 쪼개서 dataloader 구성
    data_loader = torch.utils.data.DataLoader(
        list(zip(train_x, train_y)), batch_size=16, shuffle=True
    )

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")

                # 입력과 목표값
                x, y = batch

                #input = x + x + y + y
                #target = x + y + x + y

                inputs = torch.cat((x, x, y, y), 0)
                targets = torch.cat((x, y, x, y), 0)

           
                # 예측
                outputs = model(inputs)
                #target_ouputs = model(targets)

                # concant outputs_targetoutputs & inputs_targets
                #outputs = torch.cat((outputs, target_ouputs), 0)
                #targets = torch.cat((targets, inputs), 0)
                
                # 손실 계산
                loss = -criterion(outputs, targets).mean()
                
                # 역전파
                optimizer.zero_grad()
                loss.backward()
                
                # 매개변수 업데이트
                optimizer.step()
                #scheduler.step(loss)

                # tqdm 상태 업데이트
                tepoch.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, lr: {optimizer.param_groups[0]['lr']}")


    save_model()

    
def save_model():
    global model
    torch.save(model.state_dict(), MODEL_FILENAME)
    print("Model saved to", MODEL_FILENAME)

def load_model():
    global model
    global bert_model, bert_tokenizer
    bert_model, bert_tokenizer = kobert_model()

    model = KoBertEmbedding().to(device)
    model.load_state_dict(torch.load(MODEL_FILENAME))
    model.eval()


def get_embedding(text):
    global model
    if type(text) == str:
        text = get_bert_embedding(text)
    with torch.no_grad():
        ret = model.get_latent_vector(text)
        return model.get_latent_vector(text).cpu().numpy()
    




def content_df(cached=False):
    if cached:
        print("Load cahced file")
        df = pd.read_csv("merged.csv")
        train_x = torch.load("train_x.pt")
        train_y = torch.load("train_y.pt")
        return df, train_x, train_y
    print("Loading csv files")
    file_path = '../crawling/merged.csv'


    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Content'])
    okt = konlpy.tag.Okt()
    tqdm.pandas(desc="Tokenizing Contents")
    df["Content"] = df["Content"].progress_apply(lambda x: " ".join(okt.nouns(x)))
    df = df[df["Content"].str.len() > 0]
    tqdm.pandas(desc="Vectorizing Titles")
    train_x = df["Title"].progress_apply(get_bert_embedding)
    tqdm.pandas(desc="Vectorizing Contents")


    
    train_y = df["Content"].progress_apply(get_bert_embedding)

    # save train_x train_y as file
    torch.save(train_x, "train_x.pt")
    torch.save(train_y, "train_y.pt")

    catch_path = 'merged.csv'

    # save df
    df.to_csv(catch_path, index=False)

    return df, train_x, train_y


if __name__ == '__main__':

    # if -l is specifed, load train_x, and train_y from file. else, load from csv fild and get embeding vector
    bert_model, bert_tokenizer = kobert_model()

    if "-l" in sys.argv:
        print("Loading train_x, train_y from file")
        train_x = torch.load("train_x.pt")
        train_y = torch.load("train_y.pt")
        train_model(train_x, train_y)

        sys.exit(0)
    else:
        df, train_x, train_y = content_df()
        train_model(train_x, train_y)

