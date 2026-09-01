import os
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from feature_extractor import FeatureExtractor

PRODUCTS_DIR = Path(__file__).parent / "products"
INDEX_PATH = Path(__file__).parent / "index.pkl"
BATCH_SIZE = 32


def build_index():
    extractor = FeatureExtractor()

    image_paths = sorted(PRODUCTS_DIR.glob("*.jpg"))
    if not image_paths:
        print("Nenhuma imagem encontrada em products/")
        return

    print(f"Indexando {len(image_paths)} imagens...")

    all_embeddings = []
    all_names = []

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        images = []
        names = []
        for p in batch_paths:
            try:
                img = Image.open(p)
                images.append(img)
                names.append(p.name)
            except Exception as e:
                print(f"  Erro ao abrir {p.name}: {e}")
        if images:
            embeddings = extractor.extract_batch(images)
            all_embeddings.append(embeddings)
            all_names.extend(names)
            print(f"  Processadas {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)}")

    embeddings_matrix = np.vstack(all_embeddings)

    with open(INDEX_PATH, "wb") as f:
        pickle.dump({"names": all_names, "embeddings": embeddings_matrix}, f)

    print(f"Índice salvo em {INDEX_PATH} ({len(all_names)} imagens, {embeddings_matrix.shape[1]} dims)")


if __name__ == "__main__":
    build_index()
