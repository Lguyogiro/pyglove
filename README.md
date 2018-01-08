A small Python module that allows you to train and use GloVe vectors (for more about GloVe, see https://nlp.stanford.edu/pubs/glove.pdf) in your Python appplications.

eg

```Python
>>> from pyglove import GloVe
>>> from sklearn.datasets import fetch_20newsgroups

>>> tokens = [word for mssg in fetch_20newsgroups()
...           for sent in mssg.split('\n') 
...           for word in sent.split()]

>>> model = GloVe(tokens, 
...               "20newsgroup_words.txt", # file where vectors 
...               glove_installation_dir="/path/to/glove/install", 
...               vocab_min_count=5, 
...               vector_size=50, 
...               window_size=15, 
...               run=True)
              
>>> model.most_similar("Toyota", n=3)
[('Camry', 0.72454572174443854),
 ('4Runner', 0.69299255292696094),
 ('Tercel', 0.68875869902754971)]

```