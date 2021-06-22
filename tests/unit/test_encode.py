import pytest
from jina import DocumentArray, Document
from jinahub.encoder.flair_text import FlairTextEncoder


@pytest.fixture()
def test_text():
    return 'it is a good day! the dog sits on the floor.'


def test_flairtextencoder_encode(test_text):
    encoder = FlairTextEncoder(pooling_strategy='mean')
    docs = DocumentArray([Document(text=test_text)])
    encoder.encode(docs, parameters={})

    assert docs[0].embedding.shape == (100,)
