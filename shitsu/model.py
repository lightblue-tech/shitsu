from safetensors.torch import load_model
import fasttext
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
import torch
import numpy as np
import torch
import torch.nn as nn

class FasttextEmbedRegressor(nn.Module):
    def __init__(self, input_size=300):
        super(FasttextEmbedRegressor, self).__init__()
        layer_1_size = 64
        layer_2_size = 32
        self.fc1 = nn.Linear(input_size, layer_1_size)
        self.fc2 = nn.Linear(layer_1_size, layer_2_size)
        self.fc3 = nn.Linear(layer_2_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ShitsuScorer:
    def __init__(self, lang_code):
        # Some languages do not have a crawl vector on Fasttext, so we use the wikipedia vector instead in these cases
        no_crawl_vector_langs = ["ha"]
        fasttext_repo = "ptrdvn" if lang_code in no_crawl_vector_langs else "facebook"
        
        fasttext_model_path = hf_hub_download(repo_id=f"{fasttext_repo}/fasttext-{lang_code}-vectors", filename="model.bin")
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.regressor_model = FasttextEmbedRegressor().eval()
        regressor_model_path = hf_hub_download(repo_id=f"lightblue/shitsu_text_scorer", filename=f"{lang_code}.safetensors")
        load_model(self.regressor_model, regressor_model_path)

    def score(self, text_list):
        embeddings = np.stack([self.fasttext_model.get_sentence_vector(x.replace("\n", " ")) for x in tqdm(text_list)])
        return self.regressor_model(torch.Tensor(embeddings)).detach().numpy().flatten()
