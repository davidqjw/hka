

from flashrag.utils import get_retriever, get_generator
from flashrag.prompt import PromptTemplate




class BasicPipeline:
    """Base object of all pipelines. A pipeline includes the overall process of RAG.
    If you want to implement a pipeline, you should inherit this class.
    """


    def __init__(self, config, prompt_template=None):
        self.config = config
        self.device = config["device"]
        self.retriever = None
        if prompt_template is None:
            prompt_template = PromptTemplate(config)
        self.prompt_template = prompt_template


    def run(self):
        """The overall inference process of a RAG framework."""
        pass




class SequentialPipeline(BasicPipeline):
    def __init__(self, config, prompt_template=None, retriever=None):
        """
        inference stage:
            query -> pre-retrieval -> retriever -> post-retrieval -> generator
        """
        super().__init__(config, prompt_template)
        self.generator = get_generator(config)
        if retriever is None:
            self.retriever = get_retriever(config)
        else:
            self.retriever = retriever


    def run(self, question):
        retrieval_results = self.retriever.batch_search(question)
        input_prompts = [
            self.prompt_template.get_string(question=q, retrieval_result=r)
            for q, r in zip(question, retrieval_results)
        ]
        pred_answer_list = self.generator.generate(input_prompts)


        return pred_answer_list[0] if len(pred_answer_list) == 1 else pred_answer_list
