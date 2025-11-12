import os
from pathlib import Path
import argparse
import json
import warnings
import torch
import faiss
from flashrag.retriever.utils import load_corpus
class Index_Builder:
    def __init__(
        self,
        retrieval_method,
        model_path,
        corpus_path,
        save_dir,
        max_length,
        batch_size,
        use_fp16,
        pooling_method=None,
        instruction=None,
        faiss_type=None,
        embedding_path=None,
        save_embedding=False,
        faiss_gpu=False,
        use_sentence_transformer=False,
        bm25_backend="bm25s",
        index_modal="all",
    ):


        self.retrieval_method = retrieval_method.lower()
        self.model_path = model_path
        self.corpus_path = corpus_path
        self.save_dir = save_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.instruction = instruction
        self.faiss_type = faiss_type if faiss_type is not None else "Flat"
        self.embedding_path = embedding_path
        self.save_embedding = save_embedding
        self.faiss_gpu = faiss_gpu
        self.use_sentence_transformer = use_sentence_transformer
        self.bm25_backend = bm25_backend
        self.index_modal = index_modal
        self.is_clip = ("clip" in self.retrieval_method) or (self.model_path is not None and "clip" in self.model_path)
        if pooling_method is None:
            try:
                # read pooling method from 1_Pooling/config.json
                pooling_config = json.load(open(os.path.join(self.model_path, "1_Pooling/config.json")))
                for k, v in pooling_config.items():
                    if k.startswith("pooling_mode") and v == True:
                        pooling_method = k.split("pooling_mode_")[-1]
                        if pooling_method == "mean_tokens":
                            pooling_method = "mean"
                        elif pooling_method == "cls_token":
                            pooling_method = "cls"
                        else:
                            # raise warning: not implemented pooling method
                            warnings.warn(f"Pooling method {pooling_method} is not implemented.", UserWarning)
                            pooling_method = "mean"
                        break
            except:
                print(f"Pooling method not found in {self.model_path}, use default pooling method (mean).")
                # use default pooling method
                pooling_method = "mean"
        else:
            if pooling_method not in ["mean", "cls", "pooler"]:
                raise ValueError(f"Invalid pooling method {pooling_method}.")
        self.pooling_method = pooling_method
        self.gpu_num = torch.cuda.device_count()
        # prepare save dir
        print(self.save_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        else:
            if not self._check_dir(self.save_dir):
                warnings.warn("Some files already exists in save dir and may be overwritten.", UserWarning)


        self.embedding_save_path = os.path.join(self.save_dir, f"emb_{self.retrieval_method}.memmap")


        self.corpus = load_corpus(self.corpus_path)


        print("Finish loading...")


    @staticmethod
    def _check_dir(dir_path):
        r"""Check if the dir path exists and if there is content."""


        if os.path.isdir(dir_path):
            if len(os.listdir(dir_path)) > 0:
                return False
        else:
            os.makedirs(dir_path, exist_ok=True)
        return True
   
    def build_index(self):
        self.build_dense_index()


    def encode_all(self):
        encode_data = [item["contents"] for item in self.corpus]
        if self.gpu_num > 1:
            print("Use multi gpu!")
            self.batch_size = self.batch_size * self.gpu_num
            all_embeddings = self.encoder.multi_gpu_encode(encode_data, batch_size=self.batch_size, is_query=False)
        else:
            all_embeddings = self.encoder.encode(encode_data, batch_size=self.batch_size, is_query=False)


        return all_embeddings
   
    """
    IndexBuilder class for building dense indices.
    """
    def build_dense_index(self):
        if self.use_sentence_transformer:
            from flashrag.retriever.encoder import STEncoder


            self.encoder = STEncoder(
                model_name=self.retrieval_method,
                model_path=self.model_path,
                max_length=self.max_length,
                use_fp16=self.use_fp16,
                instruction=self.instruction,
            )
            hidden_size = self.encoder.model.get_sentence_embedding_dimension()
            self.index_save_path = os.path.join(self.save_dir, f"{Path(self.corpus_path).stem}_{self.retrieval_method}_{self.faiss_type}.index")
        if self.embedding_path is not None:
            corpus_size = len(self.corpus)
            all_embeddings = self._load_embedding(self.embedding_path, corpus_size, hidden_size)
        else:
            all_embeddings = self.encode_all_clip() if self.is_clip else self.encode_all()
            if self.save_embedding:
                self._save_embedding(all_embeddings)
            del self.corpus
        self.index_save_path = os.path.join(self.save_dir, f"{Path(self.corpus_path).stem}_{self.retrieval_method}_{self.faiss_type}.index")
        if os.path.exists(self.index_save_path):
            print("The index file already exists and will be overwritten.")
        self.save_faiss_index(all_embeddings, self.faiss_type, self.index_save_path)
        print("Finish!")
       
    def save_faiss_index(
        self,
        all_embeddings,
        faiss_type,
        index_save_path,
    ):
        # build index
        print("Creating index")
        dim = all_embeddings.shape[-1]
        faiss_index = faiss.index_factory(dim, faiss_type, faiss.METRIC_INNER_PRODUCT)


        if self.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            faiss_index = faiss.index_cpu_to_all_gpus(faiss_index, co)
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)
            faiss_index = faiss.index_gpu_to_cpu(faiss_index)
        else:
            if not faiss_index.is_trained:
                faiss_index.train(all_embeddings)
            faiss_index.add(all_embeddings)


        faiss.write_index(faiss_index, index_save_path)


def main():
    parser = argparse.ArgumentParser(description="Creating index.")


    # Basic parameters
    parser.add_argument("--retrieval_method", type=str)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--corpus_path", type=str)
    parser.add_argument("--save_dir", default="indexes/", type=str)


    # Parameters for building dense index
    parser.add_argument("--max_length", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--pooling_method", type=str, default=None)
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--faiss_type", default=None, type=str)
    parser.add_argument("--embedding_path", default=None, type=str)
    parser.add_argument("--save_embedding", action="store_true", default=False)
    parser.add_argument("--faiss_gpu", default=False, action="store_true")
    parser.add_argument("--sentence_transformer", action="store_true", default=False)
    parser.add_argument("--bm25_backend", default="pyserini", choices=["bm25s", "pyserini"])


    # Parameters for build multi-modal retriever index
    parser.add_argument("--index_modal", type=str, default="all", choices=["text", "image", "all"])


    args = parser.parse_args()


    index_builder = Index_Builder(
        retrieval_method=args.retrieval_method,
        model_path=args.model_path,
        corpus_path=args.corpus_path,
        save_dir=args.save_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
        pooling_method=args.pooling_method,
        instruction=args.instruction,
        faiss_type=args.faiss_type,
        embedding_path=args.embedding_path,
        save_embedding=args.save_embedding,
        faiss_gpu=args.faiss_gpu,
        use_sentence_transformer=args.sentence_transformer,
        bm25_backend=args.bm25_backend,
        index_modal=args.index_modal,
    )
    index_builder.build_index()


if __name__ == "__main__":
    main()