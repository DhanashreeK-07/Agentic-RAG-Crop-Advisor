import pandas as pd
import faiss
import pickle
import os

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []

# 1️⃣ Load Crop Dataset
df = pd.read_csv("data/Crop_recommendation.csv")

for _, row in df.iterrows():
    text = f"""
    Soil Conditions:
    N:{row.N}
    P:{row.P}
    K:{row.K}
    Temperature:{row.temperature}
    Humidity:{row.humidity}
    pH:{row.ph}
    Rainfall:{row.rainfall}
    Crop:{row.label}
    """
    documents.append(text)

# 2️⃣ Load External Crop Documents
doc_folder = "documents"

for file in os.listdir(doc_folder):

    if file.endswith(".txt"):

        with open(os.path.join(doc_folder, file),"r",encoding="utf-8") as f:
            documents.append(f.read())

# 3️⃣ Create embeddings
embeddings = model.encode(documents)

index = faiss.IndexFlatL2(len(embeddings[0]))

index.add(embeddings)

faiss.write_index(index,"crop_vector_db.faiss")

pickle.dump(documents,open("documents.pkl","wb"))

print("Vector database created successfully")
