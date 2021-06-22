import numpy as np
import pytest
from jina import DocumentArray, Document
from jinahub.encoder.flair_text import FlairTextEncoder


@pytest.fixture()
def docs_generator():
    return DocumentArray((Document(text='random text') for _ in range(30)))


def test_flair_batch(docs_generator):
    encoder = FlairTextEncoder(pooling_strategy='mean')
    docs = docs_generator
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})

    assert len(docs.get_attributes('embedding')) == 30
    assert docs[0].embedding.shape == (100,)


def test_flair_word_encode():
    docs = []
    words = ['apple', 'banana1', 'banana2', 'studio', 'satelite', 'airplane']
    for word in words:
        docs.append(Document(text=word))

    clip_text_encoder = FlairTextEncoder()
    clip_text_encoder.encode(DocumentArray(docs), {})

    txt_to_ndarray = {}
    for d in docs:
        txt_to_ndarray[d.text] = d.embedding

    def dist(a, b):
        nonlocal txt_to_ndarray
        a_embedding = txt_to_ndarray[a]
        b_embedding = txt_to_ndarray[b]
        return np.linalg.norm(a_embedding - b_embedding)

    # assert semantic meaning is captured in the encoding
    small_distance = dist('banana1', 'banana2')
    assert small_distance < dist('banana1', 'airplane')
    assert small_distance < dist('banana1', 'satelite')
    assert small_distance < dist('banana1', 'studio')
    assert small_distance < dist('banana2', 'airplane')
