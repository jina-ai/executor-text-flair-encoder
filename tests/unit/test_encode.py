import pytest
from jina import DocumentArray, Document
from jinahub.encoder.flair_text import FlairTextEncoder


@pytest.fixture()
def docs_generator():
    return DocumentArray((Document(text='random text') for _ in range(30)))


def test_flairtextencoder_encode(docs_generator):
    encoder = FlairTextEncoder(pooling_strategy='mean')
    docs = docs_generator
    encoder.encode(docs, parameters={'batch_size': 10, 'traversal_paths': ['r']})

    assert len(docs.get_attributes('embedding')) == 30
    assert docs[0].embedding.shape == (100,)
