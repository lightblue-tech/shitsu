from datasets import load_dataset
import pandas as pd
import fasttext
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import requests
import gzip
import shutil
import os
from safetensors.torch import save_model
from huggingface_hub import hf_hub_download

from shitsu.model import FasttextEmbedRegressor

def train_regressor(X_train, X_test, y_train, y_test, train_epochs):

    # Initialize the model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = FasttextEmbedRegressor(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    batch_size = 32

    training_metrics = []

    for epoch in trange(train_epochs):
        model.train()
        train_losses = []
        for step_num, i in enumerate(trange(0, X_train.shape[0], batch_size)):
            vectors = torch.Tensor(X_train[i:i+batch_size])
            targets = torch.Tensor(y_train[i:i+batch_size])
            optimizer.zero_grad()
            outputs = model(vectors).squeeze()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss))
            if step_num % 10 == 0:
                model.eval()
                test_preds = model(torch.Tensor(X_test)).detach().numpy()
                test_mse = mean_squared_error(y_test, test_preds)
                training_metrics.append({
                    "epoch": epoch,
                    "step_num": step_num,
                    "i": i,
                    "test_mse": test_mse,
                    "train_loss": sum(train_losses) / len(train_losses),
                })
                train_losses = []
                model.train()

    return pd.DataFrame(training_metrics), model

def download_file(url, filename):
    """
    Helper method handling downloading large files from `url` to `filename`. Returns a pointer to `filename`.
    """
    chunkSize = 1024
    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
        for chunk in r.iter_content(chunk_size=chunkSize):
            if chunk: # filter out keep-alive new chunks
                pbar.update (len(chunk))
                f.write(chunk)
    return filename

get_filename = lambda x: f"cc.{x}.300.bin"

def download_fasttext_vectors(lang_code):
    filename = get_filename(lang_code)

    if os.path.isfile(filename):
        return None

    print(f"Downloading {lang_code} vectors")

    download_file(f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/{filename}.gz", f"{filename}.gz")

    print(f"Unzipping {lang_code} vectors")
    with gzip.open(f"{filename}.gz", 'rb') as f_in:
        with open(filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Removing zipped {lang_code} vectors")
    os.remove(f"{filename}.gz")

    return True

def create_quality_eval_model(lang_code, train_epochs=10):
    # Some languages do not have a crawl vector on Fasttext, so we use the wikipedia vector instead in these cases
    no_crawl_vector_langs = ["ha"]

    if lang_code in no_crawl_vector_langs:
        fasttext_model_path = hf_hub_download(repo_id=f"ptrdvn/fasttext-{lang_code}-vectors", filename="model.bin")
        fasttext_model = fasttext.load_model(fasttext_model_path)
    else:
        download_fasttext_vectors(lang_code)
        fasttext_model = fasttext.load_model(f"cc.{lang_code}.300.bin")

    dataset = load_dataset("lightblue/text_ratings", lang_code, split="train")
    text_list = dataset["selected_chunk"]
    label_float = [x / 100 for x in dataset["rating_float"]]


    embeddings = np.stack([fasttext_model.get_sentence_vector(
        x.replace("\n", " ")
        ) for x in tqdm(text_list)])

    X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
            embeddings,
            label_float,
            text_list,
            test_size=0.2,
            random_state=42
    )

    metrics_df, model = train_regressor(X_train, X_test, y_train, y_test, train_epochs)

    test_df = pd.DataFrame({
        "text": text_test,
        "gold_score": y_test,
        "pred_score": model(torch.Tensor(X_test)).detach().numpy().flatten()
    })

    save_model(model, f"{lang_code}.safetensors")

    if os.path.exists(get_filename(lang_code)):
        os.remove(get_filename(lang_code))

    return metrics_df, test_df

if __name__ == '__main__':

    langs = ['am', 'ar', 'bg', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'gu', 'ha', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'kn', 'ko', 'lt', 'mr', 'nl', 'no', 'yo', 'zh']
    
    for l in langs:
        print(l)
        metrics_df, test_df = create_quality_eval_model(l, train_epochs=5)
        print(l)
        metrics_df[["test_mse", "train_loss"]].rolling(50).mean().plot()
        plt.show()