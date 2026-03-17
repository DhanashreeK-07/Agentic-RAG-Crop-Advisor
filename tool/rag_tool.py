import faiss
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("crop_vector_db.faiss")

documents = pickle.load(open("documents.pkl","rb"))

def rag_tool(query):

    emb = model.encode([query])

    D,I = index.search(emb,4)

    results = [documents[i] for i in I[0]]

    context = "\n".join(results)

    return context
