import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss
from tqdm import tqdm
from konlpy.tag import Okt






# 2. Vectorize Korean text columns using KoBERT
class KoBertEmbedding:
    def __init__(self):
        modelname = "kykim/bert-kor-base" #'monologg/kobert'
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.model = BertModel.from_pretrained(modelname)
        self.model.eval()

    def get_embedding(self, text):
        inputs = self.tokenizer.encode_plus(
            text,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs['last_hidden_state'][:,0,:].numpy()
    

print("Loading csv files")

# Specify your file path
file_path = '../crawling/merged.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
df = df.dropna(subset=['Content'])


print("Loading KoBERT")

# Vectorize using KoBERT
kobert = KoBertEmbedding()

tqdm.pandas(desc="Vectorizing")
# Vectorizing
df["Vectors"] = df["Title"].progress_apply(kobert.get_embedding)
vectors = np.vstack(df["Vectors"].values)



print("Creating Faiss index")
# 3. Create a Faiss index and add the vectors
d = vectors.shape[1]
index = faiss.IndexFlatL2(d) 
index.add(vectors)




def search_k_nearest(text, K=10):
    vector = kobert.get_embedding(text)
    distances, indices = index.search(vector, K)
    for i in range(K):
        print(f"Rank {i+1}: {df.iloc[indices[0][i]]['Title']} (Distance: {distances[0][i]:.4f})")
    

    return pd.DataFrame([df.iloc[indices[0][i]] for i in range(K)]).drop('Vectors', axis=1).to_json(orient='records', force_ascii=False).replace('\/', '/')