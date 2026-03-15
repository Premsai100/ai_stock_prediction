from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct,VectorParams,Distance
import os
from pdf_parser import load_pdf
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from  config.settings import QDRANT_API_KEY,QDRANT_URL
from tqdm import tqdm
import uuid


embeddings = SentenceTransformer("rbhatia46/mxbai-embed-large-v1-financial-rag-matryoshka")

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
client.delete_collection("stock-fundamentals-collection")
print("✓ Deleted")

# Recreate it correctly with unnamed vectors
client.create_collection(
    collection_name="stock-fundamentals-collection",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)
print("✓ Recreated with proper unnamed 1024-dim vectors")
def chunking_text(data):
    text = data.get("text")
    symbol=data.get("symbol")
    from_year = data.get("from_year")
    to_year = data.get("to_year")
    metadata = []
    ids = []

    try:
        chunks = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=250)
        splitted_text = chunks.split_text(text)
        for i,chunk in enumerate(splitted_text):
            
            meta = {
                "symbol": symbol,
                "from": from_year,
                "to": to_year
            }
            metadata.append(meta)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{symbol}_{from_year}_{to_year}_{i}"))
            ids.append(point_id)
        return splitted_text,metadata,ids
    except Exception as error:
        print(f"this is error in chunking buddy{error}")



def embedding_text(text,batch_size):
    vector=[]
    
    with tqdm(total=len(text), desc="Embedding", unit="chunk", colour="green") as pbar:
        total_batches = (len(text) + batch_size - 1) // batch_size
        for i in range(0,len(text),batch_size):
            batch = text[i:i+batch_size]
            vec = embeddings.encode(batch,normalize_embeddings=True)
            vector.extend(vec.tolist())
            pbar.update(len(batch))
            pbar.set_postfix({"batch": f"{i // batch_size + 1}/{total_batches}"})
    return vector


def storing_embeddings(vectors, metadata, ids):
    total_batches = range(0, len(vectors), 10)
    
    with tqdm(total=len(vectors), desc="Upserting", unit="chunk", colour="blue") as pbar:
        for i in total_batches:
            batch_v   = vectors[i:i+10]
            batch_m   = metadata[i:i+10]
            batch_ids = ids[i:i+10]

            batch_points = [
                PointStruct(id=batch_ids[j], vector=batch_v[j], payload=batch_m[j])
                for j in range(len(batch_v))
            ]

            client.upsert(collection_name="stock-fundamentals-collection", points=batch_points)
            pbar.update(len(batch_v))
            pbar.set_postfix({"batch": f"{i // 10 + 1}/{(len(vectors) + 9) // 10}"})





if __name__=="__main__":
    files = os.listdir("data/raw/annual_reports/pdfs")
    
    for file in files:
            if not file.endswith(".pdf"):
                continue
            try :
                data=load_pdf(f"data/raw/annual_reports/pdfs/{file}")
                chunks,metadata,ids=chunking_text(data)
                vectors=embedding_text(chunks,80)
                storing_embeddings(vectors,metadata,ids)
            except Exception as e:
                print(f"something went wrong for {file}___{str(e)}")
                continue
        
    
       

   




