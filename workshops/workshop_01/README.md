# Deep Learning Workshop

For this first DL workshop, we propose to work on a Named Entity Recognition task.

## Requirements

* You need to have a working [Python](https://www.python.org/) environment. You can either follow the instructions
located in the `tools` directory at the root of the repository for a system-wide installation, or follow the 
[instructions](https://github.com/ArnaudFerre/AtelierDeepLearningILES/wiki) in the wiki for a local installation via
miniconda.

## Data

Two corpora will be considered for the task:
* The [NCBI Disease Corpus ](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/), a corpus of PubMed abstracts
 annotated with disease mentions and concepts.
* The [JNLPBA corpus](http://www.geniaproject.org/shared-tasks/bionlp-jnlpba-shared-task-2004), a corpus of PubMed
 abstracts annotated with bio-entity mentions

For easier usage, the NCBI Disease Corpus has been preprocessed and transformed into a tabulated format. Titles and
abstracts have been segmented, tokenized and tagged with parts of speech using 
[CoreNLP](https://stanfordnlp.github.io/CoreNLP/) and standoff annotations have been transformed into IOB format 
(B-Disease, I-Disease, O).

The JNLPBA is already in the correct format, so no preprocessing has been done.

Three approach are possible, from the simplest to the most meaningful:
1. training single word embeddings + training and testing a neural net for single word classification
2. training single word embeddings + training and testing a neural net (CNN) for contextual word classification
3. training multiword embeddings + training and testing on variable length multiword expressions (with a heuristic for deciding length)

## Helpers

We provide two functions to load the corpora. The functions will use the gensim model given as argument to map tokens to
indexes in the gensim model. If the token is not present in the gensim model, it will map it to the `#unk#` token.

For starters, two w2v models are available. You can also train yours or use randomly initialized vectors. 
Do not forget to have a `#unk#` token if you want to use the tool. 

* [s0100-w01-m128-lTrue-cTrue-i10_unk0.5.tar.gz](https://perso.limsi.fr/tourille/w2v/s0100-w01-m128-lTrue-cTrue-i10_unk0.5.tar.gz) -
 Vector size: 100 | Window: 1 | Min-count: 128
* [s0200-w08-m64-lTrue-cTrue-i10_unk0.5.tar.gz](https://perso.limsi.fr/tourille/w2v/s0200-w08-m64-lTrue-cTrue-i10_unk0.5.tar.gz) - 
 Vector size: 200 | Window: 8 | Min-count: 64
 
Both models were trained on *lowercased text* with numbers replaced with *0*.

```python
from w01pkg.ncbi import load_ncbi
from w01pkg.jnlpba import load_jnlpba

gensim_model_path = "/path/to/gensim-model.pkl"

ncbi_train_tab_data = "/path/to/data/ncbi-disease-corpus/tab-data/train.tab"
ncbi_dev_tab_data = "/path/to/data/ncbi-disease-corpus/tab-data/dev.tab"
ncbi_test_tab_data = "/path/to/data/ncbi-disease-corpus/tab-data/test.tab"

(ncbi_x_train, ncbi_y_train), (ncbi_x_dev, ncbi_y_dev), \
(ncbi_x_test, ncbi_y_test) = load_ncbi(ncbi_train_tab_data,
                                       ncbi_dev_tab_data,
                                       ncbi_test_tab_data,
                                       gensim_model_path)

jnlpba_train_tab_data = "/path/to/data/jnlpba-corpus/original-data/train/Genia4ERtask1.iob2"
jnlpba_test_tab_data = "/path/to/data/jnlpba-corpus/original-data/test/Genia4EReval1.iob2"

(jnlpba_x_train, jnlpba_y_train), (jnlpba_x_test, jnlpba_y_test) = load_jnlpba(jnlpba_train_tab_data,
                                                                               jnlpba_test_tab_data,
                                                                               gensim_model_path)
```

## Tips & Tricks

### Gensim

```python
# Load a gensim model
gensim_model = gensim.models.Word2Vec.load(gensim_model_path)

# Get vocabulary size
voc_size = len(gensim_model.wv.index2word)

# Get vector size
vector_size = gensim_model.vector_size

# Retrieve raw matrix (voc_size x vector_size)
embedding_matrix = gensim_model.wv.syn0

# Fast token looking-up in large gensim models
# The operation `x in s` is O(n) for the `list` object and O(1) for the `set` object 
# Let's build a set for token looking-up
word_set = set(gensim_model.wv.index2word)

# Storing tokens indexes into a dictionary
word_dict = {}
for i, key in enumerate(gensim_model.wv.index2word):
    word_dict[key] = i
    
# Now, if you want to fetch the id of a word, you look in the set to know if the token is in the vocabulary
# If it is the case, you fetch its id in the dictionary
if 'frite' in word_set:
    token_id = word_dict["frite"]
else:
    token_id = word_dict["#unk#"]
```