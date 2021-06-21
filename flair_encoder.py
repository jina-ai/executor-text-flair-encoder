__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

from typing import Union, Tuple, List

import numpy as np
from jina import Executor, requests


class FlairTextEncoder(Executor):
    def __init__(self,
                 embeddings: Union[Tuple[str], List[str]] = ('word:glove', ),
                 pooling_strategy: str = 'mean',
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = embeddings
        self.pooling_strategy = pooling_strategy
        self.max_length = -1  # reserved variable for future usages
        self._post_set_device = False

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
                self.logger.error(f'embedding not found: {e}')
                continue
            if emb is not None:
                embeddings_list.append(emb)
        if embeddings_list:
            from flair.embeddings import DocumentPoolEmbeddings
            self.model = DocumentPoolEmbeddings(embeddings_list, pooling=self.pooling_strategy)
            self.logger.info(f'flair encoder initialized with embeddings: {self.embeddings}')
        else:
            self.logger.error('flair encoder initialization failed.')

    @requests
    def encode(self, content: 'np.ndarray', *args, **kwargs) -> 'np.ndarray':
        """
        Encode ``Document`` content from an array of string in size `B` into a ndarray in size `B x D`.

        :param content: a 1-dimension array of string type in size `B`
        :return: an ndarray in size `B x D`
        """
        from flair.data import Sentence
        c_batch = [Sentence(row) for row in content]
        self.model.embed(c_batch)
        result = [self.tensor2array(c_text.embedding) for c_text in c_batch]
        return np.vstack(result)

    def tensor2array(self, tensor):
        if isinstance(tensor, np.ndarray):
            return tensor
        return tensor.cpu().numpy() if self.on_gpu else tensor.numpy()
