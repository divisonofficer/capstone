import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
import faiss







# 2. Vectorize Korean text columns using KoBERT
class KoBertEmbedding:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        self.model = BertModel.from_pretrained('monologg/kobert')
        self.model.eval()

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs['last_hidden_state'][:,0,:].numpy()
    

print("Loading csv files")

# Specify your file path
file_path = '../crawling/file_content_list.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
df = df.dropna(subset=['Content'])


print("Loading KoBERT")

# Vectorize using KoBERT
kobert = KoBertEmbedding()


print("Vectorizing")
#raws Content를 바탕으로 한 비교는 의미있는 뚜렷한 결과를 양산하지는 못하고 있음
#df["Vectors"] = df["Content"].apply(kobert.get_embedding)
df["Vectors"] = df["Title"].apply(kobert.get_embedding)
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