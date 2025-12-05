from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline
from flashrag.prompt import PromptTemplate
import argparse
import os
def main(queries):
    my_config = Config(
        config_file_path = 'my_config.yaml'
    )
    
    if my_config["metric_setting"]["llm_judge_setting"]["model_name"] == "openai":
      my_config["metric_setting"]["llm_judge_setting"]["openai"]["openai_setting"]["api_key"] = os.getenv("OPENAI_API_KEY")

    
    prompt_templete = PromptTemplate(
        my_config,
        system_prompt = "Answer the question based on the given document. Only give me the answer and do not output any other words.\nThe following are given documents.\n\n{reference}",
        user_prompt = "Question: {question}\nAnswer:"
    )
    pipeline = SequentialPipeline(
      my_config,
      prompt_template = prompt_templete
    )
    output = pipeline.run(queries)

    print(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, nargs='+', help='Input question(s) for the RAG pipeline')
    args = parser.parse_args()
    
    main(args.query)
 
