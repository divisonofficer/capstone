import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss
from tqdm import tqdm
from konlpy.tag import Okt

from mykobert import load_model, get_embedding, content_df



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

    df, title_vector, content_vector = content_df()
    content_vector = [x.cpu() for x in content_vector]
    print("Loading KoBERT")

    # Vectorize using KoBERT

    #train
    tqdm.pandas(desc="Train")
    
    #kobert = KoBertEmbedding()

    tqdm.pandas(desc="Vectorizing")
    # Vectorizing
    
    vectors = np.vstack(content_vector)



    print("Creating Faiss index")
    # 3. Create a Faiss index and add the vectors
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d) 
    index.add(vectors)




def search_k_nearest(text, K=10):
    #vector = kobert.get_embedding(text)
    vector = get_embedding(text)
    distances, indices = index.search(vector, K)
    for i in range(K):
        print(f"Rank {i+1}: {df.iloc[indices[0][i]]['Title']} (Distance: {distances[0][i]:.4f})")
    

    return pd.DataFrame([df.iloc[indices[0][i]] for i in range(K)]).drop('Vectors', axis=1).to_json(orient='records', force_ascii=False).replace('\/', '/')