from jina import Flow, Document, DocumentArray
from jinahub.encoder.flair_text import FlairTextEncoder


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


def test_traversal_path():
    text = 'blah'
    docs = [Document(id='root1', text=text)]
    docs[0].chunks = [Document(id='chunk11', text=text),
                      Document(id='chunk12', text=text),
                      Document(id='chunk13', text=text)
                      ]
    docs[0].chunks[0].chunks = [
        Document(id='chunk111', text=text),
        Document(id='chunk112', text=text),
    ]

    f = Flow().add(uses={
        'jtype': FlairTextEncoder.__name__,
        'with': {
            'default_traversal_paths': ['c'],
        }
    })
    with f:
        result = f.post(on='/test', inputs=docs, return_results=True)
        for path, count in [['r', 0], ['c', 3], ['cc', 0]]:
            assert len(DocumentArray(result[0].data.docs).traverse_flat([path]).get_attributes('embedding')) == count

        result = f.post(on='/test', inputs=docs, parameters={'traversal_paths': ['cc']}, return_results=True)
        for path, count in [['r', 0], ['c', 0], ['cc', 2]]:
            assert len(DocumentArray(result[0].data.docs).traverse_flat([path]).get_attributes('embedding')) == count


def test_no_documents():
    with Flow().add(uses=FlairTextEncoder) as f:
        result = f.post(on='/test', inputs=[], return_results=True)
        assert result[0].status.code == 0
