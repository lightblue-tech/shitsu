# Shitsu
<img src="https://github.com/user-attachments/assets/fd56e33d-3c3b-45f3-84b5-d70f6b8fc95d" alt="A logo of a Shit Zhu reading a book" width="300"/>


A text scorer which scores text based on the amount of useful, textbook-like information in it.
It outputs a score generally between 0 and 1 but can exceed both of these bounds as it is a regressor.

Our model is based on fasttext embeddings, meaning that it can be used on large amounts of data with limited compute quickly.

This scorer can be used to filter useful information from large text corpora in many languages.

# How to install


# How to use

### With our scorer package

```bash
pip install git+https://github.com/lightblue-tech/shitsu.git
```

```python
from shitsu import ShitsuScorer

text_list = [
    "Photosynthesis is a system of biological processes by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert light energy, typically from sunlight, into the chemical energy necessary to fuel their metabolism.",
    "Congratulations! You have all been selected to receive a free gift card worth $1000. Click on this link [Link] to claim your reward now. Limited time offer, so act fast! Don't miss out on this amazing opportunity."]

# Choose a language from one of: 'am', 'ar', 'bg', 'bn', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa', 'fi', 'fr', 'gu', 'ha', 'hi', 'hu', 'id', 'it', 'ja', 'jv', 'kn', 'ko', 'lt', 'mr', 'nl', 'no', 'yo', 'zh'
language_code = "en"
scorer = ShitsuScorer(language_code)
scores = scorer.score(text_list)
scores
# array([ 0.9897383 , -0.08109612], dtype=float32)
```

### Without our scorer package (i.e. without pip install)

```python

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
        fasttext_model_path = hf_hub_download(repo_id=f"facebook/fasttext-{lang_code}-vectors", filename="model.bin")
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
        self.regressor_model = FasttextEmbedRegressor().eval()
        regressor_model_path = hf_hub_download(repo_id=f"lightblue/shitsu_text_scorer", filename=f"{lang_code}.safetensors")
        load_model(self.regressor_model, regressor_model_path)

    def score(self, text_list):
        embeddings = np.stack([self.fasttext_model.get_sentence_vector(x.replace("\n", " ")) for x in tqdm(text_list)])
        return self.regressor_model(torch.Tensor(embeddings)).detach().numpy().flatten()

text_list = [
    "Photosynthesis is a system of biological processes by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert light energy, typically from sunlight, into the chemical energy necessary to fuel their metabolism.",
    "Congratulations! You have all been selected to receive a free gift card worth $1000. Click on this link [Link] to claim your reward now. Limited time offer, so act fast! Don't miss out on this amazing opportunity."]

scorer = ShitsuScorer("en")
scores = scorer.score(text_list)
scores
# array([ 0.9897383 , -0.08109612], dtype=float32)
```

# How we made the training data

We provided a sample of tens of thousands [MADLAD-400](https://huggingface.co/datasets/allenai/MADLAD-400) in various languages to a popular state-of-the-art LLM with the following system prompt:

```python
system_message = """You are a text filtering AI model.
Your input is a piece of text.
Your output is a score of how likely the text is to appear in a useful {language} textbook, encyclopedia, or any other important document.

Output your score on a scale of 0-100, with 0 meaning that the text contains no useful {language} information and 100 meaning that the text is very useful and is exceedingly likely to appear in a {language} textbook, encyclopedia, or any other important document. If the text is not mostly fluent, natural {language}, output 0.

Your output should be only an integer from 0-100."""
```

We then trained a small neural network on top of fasttext's embeddings to predict these scores.

We chose the languages in this dataset by making a union set of the 30 most popular languages on earth as according to [Ethnologue 2024](https://www.ethnologue.com/insights/ethnologue200/) and the 30 most popular languages within MADLAD-400.
