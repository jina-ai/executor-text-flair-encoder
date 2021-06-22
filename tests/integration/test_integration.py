from jina import Flow, Document


def data_generator(num_docs):
    for i in range(num_docs):
        doc = Document(
            text='it is a good day! the dog sits on the floor.')
        yield doc


def test_use_in_flow():
    with Flow.load_config('flow.yml') as flow:
        data = flow.post(on='/encode', inputs=data_generator(5))
        docs = data[0].docs
        for doc in docs:
            assert doc.embedding.shape == (100,)
