__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Union, Tuple, List, Any, Iterable

import numpy as np
import torch
from jina import Executor, requests, DocumentArray


def _batch_generator(data: List[Any], batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


class FlairTextEncoder(Executor):
    def __init__(self,
                 embeddings: Union[Tuple[str], List[str]] = ('word:glove',),
                 pooling_strategy: str = 'mean',
                 on_gpu: bool = False,
                 default_batch_size: int = 32,
                 default_traversal_paths: Union[str, List[str]] = 'r',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.pooling_strategy = pooling_strategy
        self.max_length = -1  # reserved variable for future usages
        self.on_gpu = on_gpu
        self.default_batch_size = default_batch_size
        self.default_traversal_paths = default_traversal_paths
        self._post_set_device = False
        self.device = torch.device('cuda:0') if self.on_gpu else torch.device('cpu')

        import flair
        flair.device = self.device
        embeddings_list = []
        for e in self.embeddings:
            model_name, model_id = e.split(':', maxsplit=1)
            emb = None
            try:
                if model_name == 'flair':
                    from flair.embeddings import FlairEmbeddings
                    emb = FlairEmbeddings(model_id)
                elif model_name == 'pooledflair':
                    from flair.embeddings import PooledFlairEmbeddings
                    emb = PooledFlairEmbeddings(model_id)
                elif model_name == 'word':
                    from flair.embeddings import WordEmbeddings
                    emb = WordEmbeddings(model_id)
                elif model_name == 'byte-pair':
                    from flair.embeddings import BytePairEmbeddings
                    emb = BytePairEmbeddings(model_id)
            except ValueError:
                # self.logger.error(f'embedding not found: {e}')
                continue
            if emb is not None:
                embeddings_list.append(emb)
        if embeddings_list:
            from flair.embeddings import DocumentPoolEmbeddings
            self.model = DocumentPoolEmbeddings(embeddings_list, pooling=self.pooling_strategy)
            # self.logger.info(f'flair encoder initialized with embeddings: {self.embeddings}')
        else:
            print('flair encoder initialization failed.')

    @requests
    def encode(self, docs: DocumentArray, parameters: dict, *args, **kwargs) -> 'np.ndarray':
        """
        Encode ``Document`` content from an array of string in size `B` into a ndarray in size `B x D`.

        :param content: a 1-dimension array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
        if docs:
            document_batches_generator = self._get_input_data(docs, parameters)
            self._create_embeddings(document_batches_generator)

    def _create_embeddings(self, document_batches_generator: Iterable):
        for document_batch in document_batches_generator:
            from flair.data import Sentence
            c_batch = [Sentence(d.text) for d in document_batch]

            self.model.embed(c_batch)
            for document, c_text in zip(document_batch, c_batch):
                document.embedding = self.tensor2array(self.tensor2array(c_text.embedding))

    def _get_input_data(self, docs: DocumentArray, parameters: dict):
        traversal_paths = parameters.get('traversal_paths', self.default_traversal_paths)
        batch_size = parameters.get('batch_size', self.default_batch_size)

        # traverse thought all documents which have to be processed
        flat_docs = docs.traverse_flat(traversal_paths)

        # filter out documents without images
        filtered_docs = [doc for doc in flat_docs if doc.text is not None]

        return _batch_generator(filtered_docs, batch_size)

    def tensor2array(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy() if self.on_gpu else tensor.numpy()
