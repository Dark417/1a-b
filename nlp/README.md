# `nlp/` — Natural Language Processing

Classic-to-neural NLP. Transformers live in their own top-level
[`../transformers/`](../transformers/) folder.

| Sub-folder | Algorithms |
|---|---|
| `text-representation/` | bag-of-words, TF-IDF, n-gram LM + smoothing |
| `embeddings/` | word2vec (CBOW, Skip-gram, **negative sampling**), GloVe, fastText |
| `language-models/` | neural LM, char-RNN |
| `seq2seq/` | encoder-decoder + attention (Bahdanau/Luong), BiLSTM-CRF tagger |

Training techniques highlighted: **negative sampling** (word2vec) and
**teacher forcing** (seq2seq).
