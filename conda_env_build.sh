#!/bin/bash

set -e

pip install -e .

pip install "numpy==1.26.4"
pip install "openai==1.67.0"
pip install "thinc==8.2.5"
pip install "spacy==3.7.5"
pip install sentence-transformers

conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0

pip install google-cloud-aiplatform
pip install vertexai
pip install replicate
pip install python-dotenv

echo "Done"
