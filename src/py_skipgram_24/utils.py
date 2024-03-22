import numpy as np

def create_input_pairs(pp_corpus, word2idx, context_size=2):
    if not isinstance(pp_corpus, list) or not all(isinstance(sentence, list) for sentence in pp_corpus):
        raise ValueError("pp_corpus must be a list of lists of strings.")
    if not isinstance(word2idx, dict) or not all(isinstance(word, str) for word in word2idx.keys()):
        raise ValueError("word2idx must be a dictionary mapping strings to integers.")
    if not isinstance(context_size, int) or context_size < 0:
        raise ValueError("context_size must be a non-negative integer.")
    idx_pairs = [(word2idx[sentence[i]], word2idx[sentence[j]])
                 for sentence in pp_corpus
                 for i in range(len(sentence))
                 for j in range(max(i - context_size, 0), min(i + context_size + 1, len(sentence)))
                 if i != j]
    return np.array(idx_pairs)

def get_vocab(tokenized_corpus):
    return list(set(token for sentence in tokenized_corpus for token in sentence))

def get_word_vectors(model, word2idx):
    embedding_weights = model.embeddings.weight.data
    word_vectors = {word: embedding_weights[idx].numpy() for word, idx in word2idx.items()}
    return word_vectors
