"""Download the models the pipeline and API load at runtime, so containers start offline.

Run at image build time (see Dockerfile). Model names match configs/pipeline.yaml.
"""
import tiktoken
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

TextEmbedding("BAAI/bge-small-en-v1.5")
TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")
tiktoken.get_encoding("cl100k_base")
print("models cached")
