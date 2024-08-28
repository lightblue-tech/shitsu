# Shitsu
<img src="https://github.com/user-attachments/assets/fd56e33d-3c3b-45f3-84b5-d70f6b8fc95d" alt="A logo of a Shit Zhu reading a book" width="300"/>

A fast multilingual text quality classifier

# Install

```bash
pip install git+https://github.com/lightblue-tech/shitsu.git
```

# Usage

```python
from shitsu import ShitsuScorer

scorer = ShitsuScorer("ja")

scorer.score(["こんにちは"])
```
