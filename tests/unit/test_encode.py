import os

import numpy as np
from jina import DocumentArray, Document
from jinahub.encoder.flair_encoder import FlairTextEncoder

directory = os.path.dirname(os.path.realpath(__file__))

docs = DocumentArray([Document(text='it is a good day!, the dog sits on the floor.')])


def test_flairtextencoder_encode():
    encoder = FlairTextEncoder(pooling_strategy='mean')
    embeddings = {}

    docs = DocumentArray([Document(text='it is a good day!, the dog sits on the floor.')])
    encoder.encode(docs, parameters={})
    assert docs[0].embedding.shape == (100,)
