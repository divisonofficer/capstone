import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import sys
import konlpy

device = torch.device("cuda")

MODEL_FILENAME = "mymodel.pt"

def kobert_model():
    modelname = "kykim/bert-kor-base" #'monologg/kobert'
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


class KoBertEmbedding(nn.Module):
    def __init__(self):
        super(KoBertEmbedding, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.do = nn.Dropout(0.1)

    
    def forward(self, x):
        x = self.fc(x)
        x = self.do(x)
        return x
    

    

num_epochs = 100
model = None

def train_model(train_x, train_y):
    global model
    model = KoBertEmbedding().to(device)
    # 모델 인스턴스 생성 및 GPU로 이동


    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CosineSimilarity(dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # 학습 루프
    for epoch in range(num_epochs):
        model.train()
        # get batch from train x and train_y
        print(f"Epoch {epoch+1}/{num_epochs}")
        inputs = list(zip(train_x, train_y))

        
        
        for title_features, content_features in tqdm(inputs, desc="Training"):
         # dataloader는 Title_Features와 Content_Features를 제공
     

            # 순전파
            title_output = model(title_features)
            content_output = model(content_features)

            # 손실 계산
            loss = -criterion(title_output, content_output).mean()  # 코사인 유사도를 최대화

            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
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
    text = get_bert_embedding(text)
    with torch.no_grad():
        return model.forward(text).cpu().numpy()
    




def content_df():
    print("Loading csv files")
    file_path = '../crawling/merged.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Content'])
    okt = konlpy.tag.Okt()
    tqdm.pandas(desc="Tokenizing Contents")
    df["Content"] = df["Content"].progress_apply(lambda x: " ".join(okt.nouns(x)))

    tqdm.pandas(desc="Vectorizing Titles")
    train_x = df["Title"].progress_apply(get_bert_embedding)
    tqdm.pandas(desc="Vectorizing Contents")

    df = df[df["Content"].str.len() > 0]
    
    train_y = df["Content"].progress_apply(get_bert_embedding)

    # save train_x train_y as file
    torch.save(train_x, "train_x.pt")
    torch.save(train_y, "train_y.pt")

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

