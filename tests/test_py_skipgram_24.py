from py_skipgram_24 import SkipgramModel, create_input_pairs, get_vocab, MyPreprocessor, train_model, get_word_vectors

def test_get_vocab():
    tokenized_corpus = [["this", "is", "a", "sentence"], ["this", "is", "another", "sentence"]]
    vocab = get_vocab(tokenized_corpus)
    assert set(vocab) == set(["this", "is", "a", "sentence", "another"]), "Vocabulary does not match expected result"

def test_SkipgramModel():
    # Test that vocab_size must be a positive integer
    try:
        SkipgramModel(-1, 100)
    except ValueError:
        assert True
    else:
        assert False

    # Test that embedding_dim must be a positive integer
    try:
        SkipgramModel(100, -1)
    except ValueError:
        assert True
    else:
        assert False

def test_create_input_pairs_pp_corpus_type():
    pp_corpus = "not a list of lists"
    word2idx = {"this": 0, "is": 1, "a": 2, "sentence": 3, "another": 4}

    try:
        create_input_pairs(pp_corpus, word2idx)
    except ValueError:
        assert True
    else:
        assert False

def test_create_input_pairs_word2idx_type():
    pp_corpus = [["this", "is", "a", "sentence"], ["this", "is", "another", "sentence"]]
    word2idx = "not a dictionary"

    try:
        create_input_pairs(pp_corpus, word2idx)
    except ValueError:
        assert True
    else:
        assert False

def test_create_input_pairs_context_size_value():
    pp_corpus = [["this", "is", "a", "sentence"], ["this", "is", "another", "sentence"]]
    word2idx = {"this": 0, "is": 1, "a": 2, "sentence": 3, "another": 4}

    try:
        create_input_pairs(pp_corpus, word2idx, -1)
    except ValueError:
        assert True
    else:
        assert False

def test_get_vocab():
    # Test that tokenized_corpus must be a list of lists of strings
    try:
        get_vocab("not a list of lists")
    except ValueError:
        assert True
    else:
        assert False

def test_get_word_vectors():
    model = SkipgramModel(10, 100)  # Assuming SkipgramModel is defined
    word2idx = {"this": 0, "is": 1, "a": 2, "sentence": 3, "another": 4}

    # Test that word2idx must be a dictionary mapping strings to integers
    try:
        get_word_vectors(model, "not a dictionary")
    except ValueError:
        assert True
    else:
        assert False

    # Test that model must have 'embeddings' with 'weight' attribute
    try:
        get_word_vectors("not a model", word2idx)
    except ValueError:
        assert True
    else:
        assert False

def test_SkipgramModel_init():
    # Test that embedding_dim must be a positive integer
    try:
        SkipgramModel(100, 0)
    except ValueError:
        assert True
    else:
        assert False

def test_create_input_pairs_context_size_type():
    pp_corpus = [["this", "is", "a", "sentence"], ["this", "is", "another", "sentence"]]
    word2idx = {"this": 0, "is": 1, "a": 2, "sentence": 3, "another": 4}

    # Test that context_size must be an integer
    try:
        create_input_pairs(pp_corpus, word2idx, "not an integer")
    except ValueError:
        assert True
    else:
        assert False

def test_get_vocab_input_type():
    # Test that tokenized_corpus must be a list of lists of strings
    try:
        get_vocab("not a list of lists")
    except ValueError:
        assert True
    else:
        assert False

def test_get_word_vectors_model_type():
    model = "not a model"  # Assuming SkipgramModel is defined
    word2idx = {"this": 0, "is": 1, "a": 2, "sentence": 3, "another": 4}

    # Test that model must be an instance of SkipgramModel
    try:
        get_word_vectors(model, word2idx)
    except ValueError:
        assert True
    else:
        assert False

