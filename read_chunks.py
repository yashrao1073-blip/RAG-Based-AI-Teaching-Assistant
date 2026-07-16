import requests
import os
import json
import joblib
import pandas as pd

BATCH_SIZE = 64


def create_embedding(text_list):
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    if r.status_code != 200:
        print("Status Code:", r.status_code)
        print("Response:", r.text)
        raise Exception("Embedding request failed.")

    return r.json()["embeddings"]


jsons = os.listdir("jsons")

my_dicts = []
chunk_id = 0

for json_file in jsons:

    with open(f"jsons/{json_file}", "r", encoding="utf-8") as f:
        content = json.load(f)

    chunks = content["chunks"]

    # Process chunks in batches
    for start in range(0, len(chunks), BATCH_SIZE):

        batch_chunks = chunks[start:start + BATCH_SIZE]
        batch_texts = [chunk["text"] for chunk in batch_chunks]

        embeddings = create_embedding(batch_texts)

        for chunk, embedding in zip(batch_chunks, embeddings):
            chunk["chunk_id"] = chunk_id
            chunk["embedding"] = embedding

            my_dicts.append(chunk)
            chunk_id += 1

        print(
            f"{json_file}: Processed {min(start + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks"
        )

print(f"\nFinished! Total chunks embedded: {len(my_dicts)}")

df = pd.DataFrame.from_records(my_dicts)
# Save this dataframe
joblib.dump(df, 'embeddings.joblib')