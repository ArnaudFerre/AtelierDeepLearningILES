# Deep Learning Workshop

For this first DL workshop, we propose to work on a Named Entity Recognition task.

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