import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss
from tqdm import tqdm
from konlpy.tag import Okt

from mykobert import load_model, get_embedding, get_bert_embedding, content_df, HIDDEN_SIZE



device = torch.device("cuda")

index = None
# 2. Vectorize Korean text columns using KoBERT
class KoBertEmbedding:
    def __init__(self):
        modelname = "kykim/bert-kor-base" #'monologg/kobert'
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.model = BertModel.from_pretrained(modelname)
        self.model.to(device)
        self.model.eval()

    def get_embedding(self, text):
        # if text is short, repeat it until it reaches 512 tokens
        if len(text) < 512:
            text = (text + " ") * (512 // len(text) + 1)

        inputs = self.tokenizer.encode_plus(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        inputs.to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs['last_hidden_state'][:,0,:].cpu().numpy()
    



# Read the CSV file into a DataFrame
df = None


def init_faiss():
    global index, df
    
    print("Loading csv files")
    load_model()

    df, title_vector, content_vector = content_df(cached=True)
    title_vector = title_vector.apply(lambda x: x.cpu().numpy())
    content_vector = content_vector.apply(lambda x: x.cpu().numpy())
    tqdm.pandas(desc="Vectorizing")
    

    #-- 추가 레이어 적용할 시

    #content_vector = content_vector.progress_apply(lambda x: get_embedding(x))
    #title_vector = title_vector.progress_apply(lambda x: get_embedding(x))
    
    #reshape from (n, ) to (n, 512), get_embedding returns (1,512)
    
    #content_vector = np.array(content_vector.tolist()).reshape(-1, HIDDEN_SIZE)
    #title_vector = np.array(title_vector.tolist()).reshape(-1, HIDDEN_SIZE)

    print(content_vector.shape, title_vector.shape)
    content_vector = np.array(content_vector.tolist()).reshape(-1, 768)
    title_vector = np.array(title_vector.tolist()).reshape(-1, 768)



    print("Loading KoBERT")

    # Vectorize using KoBERT

    #train
    tqdm.pandas(desc="Train")
    
    #kobert = KoBertEmbedding()

    tqdm.pandas(desc="Vectorizing")
    # Vectorizing

   # 새로운 배열 생성
    print(title_vector.shape)

    #vectors = np.empty((title_vector.shape[0] * 2, HIDDEN_SIZE), dtype=title_vector.dtype)
    vectors = np.empty((title_vector.shape[0] * 2, 768), dtype=title_vector.dtype)

    # 각 행 번갈아 가며 추가
    vectors[0::2] = title_vector
    vectors[1::2] = content_vector


    print("Creating Faiss index")
    # 3. Create a Faiss index and add the vectors
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d) 
    index.add(vectors)
    print("Faiss index created, shape:", d)




def search_k_nearest(text, K=10):
    vector = get_bert_embedding(text)
    #주가 레이어 사용 시
    #vector = get_embedding(text)
    
    #if vector is cuda, convert to cpu.numpy
    if type(vector) == torch.Tensor:
        vector = vector.cpu().numpy()

    distances, indices = index.search(vector, K)
    for i in range(K):
        print(f"Rank {i+1}: {df.iloc[int(indices[0][i] / 2)]['Title']} (Distance: {distances[0][i]:.4f})")
    

    return pd.DataFrame([df.iloc[int(indices[0][i] / 2)] for i in range(K)]).to_json(orient='records', force_ascii=False).replace('\/', '/')