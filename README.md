![image](https://github.com/mytechnotalent/rea/blob/main/Reverse%20Engineering%20Assistant.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant
Reverse Engineering Assistant utilizing RAG and an LLM.

Retrieval-Augmented Generation (RAG) is a powerful technique in natural language processing (NLP) that combines retrieval-based methods with generative models to produce more accurate and contextually relevant outputs. This approach was introduced in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Facebook AI Research (FAIR).

For further reading and a deeper understanding of RAG, refer to the original paper by Facebook AI Research: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401). 

## Overview of the RAG Architecture

The RAG model consists of two main components:
1. **Retriever**: This component retrieves relevant documents from a large corpus based on the input query.
2. **Generator**: This component generates responses conditioned on the retrieved documents.

## Mathematical Formulation

### Retriever

The retriever selects the top $k$ documents from a corpus $\mathcal{D}$ based on their relevance to the input query $q$.

### Generator

The generator produces a response $r$ based on the input query $q$ and the retrieved documents $\{d_1, d_2, \ldots, d_k\}$.

### Combining Retriever and Generator

The final probability of generating a response $r$ given the query $q$ is obtained by marginalizing over the top $k$ retrieved documents:

$$
P(r \mid q) = \sum_{i=1}^{k} P(d_i \mid q) P(r \mid q, d_i)
$$

Here, $P(d_i \mid q)$ is the normalized relevance score of document $d_i$ given the query $q$, and $P(r \mid q, d_i)$ is the probability of generating response $r$ given the query $q$ and document $d_i$.

## Implementation Details

### Training

The RAG model is trained in two stages:
1. **Retriever Training**: The retriever is trained to maximize the relevance score $s(q, d_i)$ for relevant documents.
2. **Generator Training**: The generator is trained to maximize the probability $P(r \mid q, d_i)$ for the ground-truth responses.

### Inference

During inference, the RAG model retrieves the top $k$ documents for a given query and generates a response conditioned on these documents. The final response is obtained by marginalizing over the retrieved documents as described above.

## Conclusion

RAG leverages the strengths of both retrieval-based and generation-based models to produce more accurate and informative responses. By conditioning the generation on retrieved documents, RAG can incorporate external knowledge from large corpora, leading to better performance on various tasks.

The combination of retriever and generator in the RAG model makes it a powerful approach for tasks that require access to external knowledge and the ability to generate coherent and contextually appropriate responses.

### Install Packages


```python
%pip install llama-index==0.10.18 llama-index-llms-groq==0.1.3 groq==0.4.2 llama-index-embeddings-huggingface==0.2.0
```

    Requirement already satisfied: llama-index==0.10.18 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.10.18)
    Requirement already satisfied: llama-index-llms-groq==0.1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.1.3)
    Requirement already satisfied: groq==0.4.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.4.2)
    Requirement already satisfied: llama-index-embeddings-huggingface==0.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.2.0)
    Requirement already satisfied: llama-index-agent-openai<0.2.0,>=0.1.4 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.7)
    Requirement already satisfied: llama-index-cli<0.2.0,>=0.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.13)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.18 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.10.59)
    Requirement already satisfied: llama-index-embeddings-openai<0.2.0,>=0.1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.11)
    Requirement already satisfied: llama-index-indices-managed-llama-cloud<0.2.0,>=0.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.6)
    Requirement already satisfied: llama-index-legacy<0.10.0,>=0.9.48 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.9.48)
    Requirement already satisfied: llama-index-llms-openai<0.2.0,>=0.1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.27)
    Requirement already satisfied: llama-index-multi-modal-llms-openai<0.2.0,>=0.1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.8)
    Requirement already satisfied: llama-index-program-openai<0.2.0,>=0.1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.7)
    Requirement already satisfied: llama-index-question-gen-openai<0.2.0,>=0.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.3)
    Requirement already satisfied: llama-index-readers-file<0.2.0,>=0.1.4 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.32)
    Requirement already satisfied: llama-index-readers-llama-parse<0.2.0,>=0.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index==0.10.18) (0.1.6)
    Requirement already satisfied: llama-index-llms-openai-like<0.2.0,>=0.1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-llms-groq==0.1.3) (0.1.3)
    Requirement already satisfied: anyio<5,>=3.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq==0.4.2) (4.4.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq==0.4.2) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq==0.4.2) (0.27.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq==0.4.2) (2.8.2)
    Requirement already satisfied: sniffio in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq==0.4.2) (1.3.1)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq==0.4.2) (4.12.2)
    Requirement already satisfied: huggingface-hub>=0.19.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (0.24.5)
    Requirement already satisfied: sentence-transformers<3.0.0,>=2.6.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-embeddings-huggingface==0.2.0) (2.7.0)
    Requirement already satisfied: idna>=2.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5,>=3.5.0->groq==0.4.2) (3.7)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx<1,>=0.23.0->groq==0.4.2) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx<1,>=0.23.0->groq==0.4.2) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->groq==0.4.2) (0.14.0)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (3.15.4)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (2024.6.1)
    Requirement already satisfied: packaging>=20.9 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (6.0.1)
    Requirement already satisfied: requests in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (2.32.3)
    Requirement already satisfied: tqdm>=4.42.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (4.66.4)
    Requirement already satisfied: aiohttp in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (3.10.0)
    Requirement already satisfied: minijinja>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (2.0.1)
    Requirement already satisfied: SQLAlchemy>=1.4.49 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (2.0.31)
    Requirement already satisfied: dataclasses-json in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (0.6.7)
    Requirement already satisfied: deprecated>=1.2.9.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.2.14)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.0.8)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (3.3)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (3.8.1)
    Requirement already satisfied: numpy<2.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.26.4)
    Requirement already satisfied: openai>=1.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.38.0)
    Requirement already satisfied: pandas in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (2.2.2)
    Requirement already satisfied: pillow>=9.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (10.4.0)
    Requirement already satisfied: tenacity!=8.4.0,<9.0.0,>=8.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (8.5.0)
    Requirement already satisfied: tiktoken>=0.3.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (0.7.0)
    Requirement already satisfied: typing-inspect>=0.8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (0.9.0)
    Requirement already satisfied: wrapt in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.16.0)
    Requirement already satisfied: llamaindex-py-client<0.2.0,>=0.1.19 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-indices-managed-llama-cloud<0.2.0,>=0.1.2->llama-index==0.10.18) (0.1.19)
    Requirement already satisfied: transformers<5.0.0,>=4.37.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq==0.1.3) (4.43.3)
    Requirement already satisfied: beautifulsoup4<5.0.0,>=4.12.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index==0.10.18) (4.12.3)
    Requirement already satisfied: pypdf<5.0.0,>=4.0.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index==0.10.18) (4.3.1)
    Requirement already satisfied: striprtf<0.0.27,>=0.0.26 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index==0.10.18) (0.0.26)
    Requirement already satisfied: llama-parse>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-readers-llama-parse<0.2.0,>=0.1.2->llama-index==0.10.18) (0.4.9)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->groq==0.4.2) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->groq==0.4.2) (2.20.1)
    Requirement already satisfied: torch>=1.11.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (2.4.0)
    Requirement already satisfied: scikit-learn in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (1.5.1)
    Requirement already satisfied: scipy in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (1.14.0)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (2.3.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (23.2.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (1.9.4)
    Requirement already satisfied: soupsieve>1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from beautifulsoup4<5.0.0,>=4.12.3->llama-index-readers-file<0.2.0,>=0.1.4->llama-index==0.10.18) (2.5)
    Requirement already satisfied: click in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (8.1.7)
    Requirement already satisfied: joblib in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (2024.7.24)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface==0.2.0) (2.2.2)
    Requirement already satisfied: greenlet!=0.4.17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (3.0.3)
    Requirement already satisfied: sympy in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (1.13.1)
    Requirement already satisfied: jinja2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (3.1.4)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq==0.1.3) (0.4.3)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq==0.1.3) (0.19.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (3.21.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (2.9.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (2024.1)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from scikit-learn->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (3.5.0)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-core<0.11.0,>=0.10.18->llama-index==0.10.18) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from jinja2->torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from sympy->torch>=1.11.0->sentence-transformers<3.0.0,>=2.6.1->llama-index-embeddings-huggingface==0.2.0) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.


### Import Libraries


```python
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
# import os
# from dotenv import load_dotenv
# load_dotenv()
import warnings
warnings.filterwarnings('ignore')
```

### ACTION ITEM
Visit https://console.groq.com/keys and set up an API Key then replace `<GROQ_API_KEY>` below with the newly generated key.


```python
import os

os.environ["GROQ_API_KEY"] = "<GROQ_API_KEY>"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

### Data Ingestion


```python
reader = SimpleDirectoryReader(input_files=["files/reversing-for-everyone.pdf"])
documents = reader.load_data()
```


```python
len(documents)
```




    430




```python
documents[4].metadata
```




    {'page_label': '5',
     'file_name': 'reversing-for-everyone.pdf',
     'file_path': 'files/reversing-for-everyone.pdf',
     'file_type': 'application/pdf',
     'file_size': 25112355,
     'creation_date': '2024-08-02',
     'last_modified_date': '2024-08-02'}



### Chunking


```python
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(documents, show_progress=True)
```


    Parsing nodes:   0%|          | 0/430 [00:00<?, ?it/s]



```python
len(nodes)
```




    430




```python
nodes[0].metadata
```




    {'page_label': '1',
     'file_name': 'reversing-for-everyone.pdf',
     'file_path': 'files/reversing-for-everyone.pdf',
     'file_type': 'application/pdf',
     'file_size': 25112355,
     'creation_date': '2024-08-02',
     'last_modified_date': '2024-08-02'}



### Embedding Model


```python
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
```

### Define LLM Model


```python
llm = Groq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY)
```

### Configure Service Context


```python
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)
```

### Create Vector Store Index


```python
vector_index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=service_context, node_parser=nodes)
```


    Parsing nodes:   0%|          | 0/430 [00:00<?, ?it/s]



    Generating embeddings:   0%|          | 0/430 [00:00<?, ?it/s]


#### Persist/Save Index


```python
vector_index.storage_context.persist(persist_dir="./storage_mini")
```

#### Define Storage Context


```python
storage_context = StorageContext.from_defaults(persist_dir="./storage_mini")
```

#### Load Index


```python
index = load_index_from_storage(storage_context, service_context=service_context)
```

### Define Query Engine


```python
query_engine = index.as_query_engine(service_context=service_context)
```

#### Feed in user query


```python
import gradio as gr


def query_function(query):
    """
    Processes a query using the query engine and returns the response.

    Args:
        query (str): The query string to be processed by the query engine.

    Returns:
        str: The response generated by the query engine based on the input query.
        
    Example:
        >>> query_function("What is Reverse Engineering?")
        'Reverse engineering is the process of deconstructing an object to understand its design, architecture, and functionality.'
    """
    response = query_engine.query(query)
    return response


iface = gr.Interface(
    fn=query_function,                  # Function to call
    inputs=gr.Textbox(label="Query"),   # Input component
    outputs=gr.Textbox(label="Response") # Output component
)

iface.launch()
```

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.

![image](https://github.com/mytechnotalent/rea/blob/main/example.png?raw=true)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
