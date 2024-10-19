import pandas as pd
from sentence_transformers import SentenceTransformer

data = pd.read_csv("data.csv")

text = list(data["PNT_ATRISKNOTES_TX"])

model = SentenceTransformer("all-mpnet-base-v2")
sentences = [
                "Something fell from a great height",
                "Someone fell from a great height",
                "An arc flash was possible",
                "An arc flash happened",
                "there were cars driving quickly nearby",
                "there was traffic nearby",
                "there was exposed equipment",
                "there was exposed machinery",
                "there was exposure to very hot objects",
                "there was the possibility of very high temperatures",
                "someone could have been exposed to steam",
                "steam was released in an uncontrolled environment",
                "fire was exposed in a dangerous way",
                "there was an explosion",
                "there was a possibility of an explosion",
                "there could have been exposure to high pressure",
                "there was exposed electricity",
                "there was exposed toxic chemicals",
                "unsafe workplace practices were observed"              
            ]
embedding = model.encode(sentences)
embeddings = model.encode(text)
similarities = model.similarity(embedding, embeddings).sum(dim=0).numpy().tolist()
df = pd.DataFrame({'Text': text, 'similarity': similarities})
df_sorted = df.sort_values(by='similarity')
df_sorted.to_csv("similarity.csv", index=False)