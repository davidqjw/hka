import os
from flashrag.retriever.utils import load_corpus, load_docs, convert_numpy
import faiss
import json
from flashrag.retriever.encoder import Encoder, STEncoder
import warnings
class BaseRetriever:
    """Base object for all retrievers."""

    def __init__(self, config):
        self._config = config
        self.update_config()

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, config_data):
        self._config = config_data
        self.update_config()

    def update_config(self):
        self.update_base_setting()
        self.update_additional_setting()

    def update_base_setting(self):
        self.retrieval_method = self._config["retrieval_method"]
        self.topk = self._config["retrieval_topk"]

        self.index_path = self._config["index_path"]
        self.corpus_path = self._config["corpus_path"]

        self.save_cache = self._config["save_retrieval_cache"]
        self.use_cache = self._config["use_retrieval_cache"]
        self.cache_path = self._config["retrieval_cache_path"]

        if self.save_cache:
            self.cache_save_path = os.path.join(self._config["save_dir"], "retrieval_cache.json")
            self.cache = {}
        if self.use_cache:
            assert self.cache_path is not None
            with open(self.cache_path, "r") as f:
                self.cache = json.load(f)
        self.silent = self._config["silent_retrieval"] if "silent_retrieval" in self._config else False

    def update_additional_setting(self):
        pass

    def _save_cache(self):
        self.cache = convert_numpy(self.cache)

        def custom_serializer(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")

        with open(self.cache_save_path, "w") as f:
            json.dump(self.cache, f, indent=4, default=custom_serializer)

    def _search(self, query: str, num: int, return_score: bool) -> List[Dict[str, str]]:
        r"""Retrieve topk relevant documents in corpus.

        Return:
            list: contains information related to the document, including:
                contents: used for building index
                title: (if provided)
                text: (if provided)

        """

        pass

    def _batch_search(self, query, num, return_score):
        pass

    def search(self, *args, **kwargs):
        return self._search(*args, **kwargs)

    def batch_search(self, *args, **kwargs):
        return self._batch_search(*args, **kwargs)
    
class DenseRetriever(BaseTextRetriever):
    r"""Dense retriever based on pre-built faiss index."""

    def __init__(self, config: dict, corpus=None):
        super().__init__(config)

        self.load_corpus(corpus)
        self.load_index()
        self.load_model()

    def load_corpus(self, corpus):
        if corpus is None:
            self.corpus = load_corpus(self.corpus_path)
        else:
            self.corpus = corpus

    def load_index(self):
        if self.index_path is None or not os.path.exists(self.index_path):
            raise Warning(f"Index file {self.index_path} does not exist!")
        self.index = faiss.read_index(self.index_path)
        if self.use_faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

    def update_additional_setting(self):
        self.query_max_length = self._config["retrieval_query_max_length"]
        self.pooling_method = self._config["retrieval_pooling_method"]
        self.use_fp16 = self._config["retrieval_use_fp16"]
        self.batch_size = self._config["retrieval_batch_size"]
        self.instruction = self._config["instruction"]

        self.retrieval_model_path = self._config["retrieval_model_path"]
        self.use_st = self._config["use_sentence_transformer"]
        self.use_faiss_gpu = self._config["faiss_gpu"]

    def load_model(self):
        if self.use_st:
            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self._config["retrieval_model_path"],
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
                silent=self.silent,
            )
        else:
            # check pooling method
            self._check_pooling_method(self.retrieval_model_path, self.pooling_method)
            self.encoder = Encoder(
                model_name=self.retrieval_method,
                model_path=self.retrieval_model_path,
                pooling_method=self.pooling_method,
                max_length=self.query_max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
            )

    def _check_pooling_method(self, model_path, pooling_method):
        try:
            # read pooling method from 1_Pooling/config.json
            pooling_config = json.load(open(os.path.join(model_path, "1_Pooling/config.json")))
            for k, v in pooling_config.items():
                if k.startswith("pooling_mode") and v == True:
                    detect_pooling_method = k.split("pooling_mode_")[-1]
                    if detect_pooling_method == "mean_tokens":
                        detect_pooling_method = "mean"
                    elif detect_pooling_method == "cls_token":
                        detect_pooling_method = "cls"
                    else:
                        # raise warning: not implemented pooling method
                        warnings.warn(f"Pooling method {detect_pooling_method} is not implemented.", UserWarning)
                        detect_pooling_method = "mean"
                    break
        except:
            detect_pooling_method = None

        if detect_pooling_method is not None and detect_pooling_method != pooling_method:
            warnings.warn(
                f"Pooling method in model config file is {detect_pooling_method}, but the input is {pooling_method}. Please check carefully."
            )

    def _search(self, query: str, num: int = None, return_score=False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        scores = scores.tolist()
        idxs = idxs[0]
        scores = scores[0]

        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query: List[str], num: int = None, return_score=False):
        if isinstance(query, str):
            query = [query]
        if num is None:
            num = self.topk
        batch_size = self.batch_size

        results = []
        scores = []
        emb = self.encoder.encode(query, batch_size=batch_size, is_query=True)
        scores, idxs = self.index.search(emb, k=num)
        scores = scores.tolist()
        idxs = idxs.tolist()

        flat_idxs = sum(idxs, [])
        results = load_docs(self.corpus, flat_idxs)
        results = [results[i * num : (i + 1) * num] for i in range(len(idxs))]

        if return_score:
            return results, scores
        else:
            return results