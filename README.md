A small Python module that allows you to train and use GloVe vectors in your Python appplications. For more about GloVe, see https://nlp.stanford.edu/pubs/glove.pdf

eg

```Python
>>> from pyglove import GloVe
>>> from sklearn.datasets import fetch_20newsgroups

>>> sentences = [sentence.split() for sentence in fetch_20newsgroups()['data']]
>>> glove_pth = "/path/to/glove/install"
>>> vector_file_name = "20newsgroup_words.txt"  # file where vectors will be stored

>>> model = GloVe(sentences, vector_file_name, glove_installation_dir=glove_pth, 
                  vocab_min_count=5, vector_size=50, window_size=15)
              
>>> model.most_similar("Camry", n=1)
[('Toyota', 0.69786818922200133)]

```
