![image](https://github.com/mytechnotalent/rea/blob/main/Reverse%20Engineering%20Assistant.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant
A Reverse Engineering Assistant leveraging Retrieval-Augmented Generation (RAG) and the LLaMA-3.1-8B-Instant Large Language Model (LLM). This tool is designed to revolutionize reverse engineering tasks by combining machine learning with retrieval-based systems.

## Origin of the RAG Architecture

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

### Install Conda Environment
1. To select a Conda environment in Visual Studio Code, press the play button in the next cell which will open up a command prompt then select `Python Environments...`
2. A new command prompt will pop up and select `+ Create Python Environment`.
3. A new command prompt will again pop up and select `Conda Creates a .conda Conda environment in the current workspace`.
4. A new command prompt will again pop up and select `* Python 3.11`.


```python
!conda create -n rea python=3.11 -y
```

    Channels:
     - defaults
    Platform: osx-arm64
    Collecting package metadata (repodata.json): done
    Solving environment: done
    
    ## Package Plan ##
    
      environment location: /opt/anaconda3/envs/rea
    
      added / updated specs:
        - python=3.11
    
    
    The following NEW packages will be INSTALLED:
    
      bzip2              pkgs/main/osx-arm64::bzip2-1.0.8-h80987f9_6 
      ca-certificates    pkgs/main/osx-arm64::ca-certificates-2024.7.2-hca03da5_0 
      libffi             pkgs/main/osx-arm64::libffi-3.4.4-hca03da5_1 
      ncurses            pkgs/main/osx-arm64::ncurses-6.4-h313beb8_0 
      openssl            pkgs/main/osx-arm64::openssl-3.0.14-h80987f9_0 
      pip                pkgs/main/osx-arm64::pip-24.0-py311hca03da5_0 
      python             pkgs/main/osx-arm64::python-3.11.9-hb885b13_0 
      readline           pkgs/main/osx-arm64::readline-8.2-h1a28f6b_0 
      setuptools         pkgs/main/osx-arm64::setuptools-69.5.1-py311hca03da5_0 
      sqlite             pkgs/main/osx-arm64::sqlite-3.45.3-h80987f9_0 
      tk                 pkgs/main/osx-arm64::tk-8.6.14-h6ba3021_0 
      tzdata             pkgs/main/noarch::tzdata-2024a-h04d1e81_0 
      wheel              pkgs/main/osx-arm64::wheel-0.43.0-py311hca03da5_0 
      xz                 pkgs/main/osx-arm64::xz-5.4.6-h80987f9_1 
      zlib               pkgs/main/osx-arm64::zlib-1.2.13-h18a0788_1 
    
    
    
    Downloading and Extracting Packages:
    
    Preparing transaction: done
    Verifying transaction: done
    Executing transaction: done
    #
    # To activate this environment, use
    #
    #     $ conda activate rea
    #
    # To deactivate an active environment, use
    #
    #     $ conda deactivate
    


### !!! ACTION ITEM !!!
In order for the Conda environment to be available, you need to close down VSCode and reload it and select `rea` in the Kernel area in the top-right of VSCode.
1. In the VSCode pop-up command window select `Select Another Kernel...`.
2. In the next command window select `Python Environments...`.
3. In the next command window select `rea (Python 3.11.9)`.

### Install Packages


```python
%pip install ipywidgets 
%pip install llama-index 
%pip install llama-index-embeddings-huggingface 
%pip install llama-index-llms-groq 
%pip install groq 
%pip install gradio
```

    Collecting ipywidgets
      Using cached ipywidgets-8.1.3-py3-none-any.whl.metadata (2.4 kB)
    Requirement already satisfied: comm>=0.1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (0.2.2)
    Requirement already satisfied: ipython>=6.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (8.26.0)
    Requirement already satisfied: traitlets>=4.3.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (5.14.3)
    Collecting widgetsnbextension~=4.0.11 (from ipywidgets)
      Using cached widgetsnbextension-4.0.11-py3-none-any.whl.metadata (1.6 kB)
    Collecting jupyterlab-widgets~=3.0.11 (from ipywidgets)
      Using cached jupyterlab_widgets-3.0.11-py3-none-any.whl.metadata (4.1 kB)
    Requirement already satisfied: decorator in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (0.1.7)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (3.0.47)
    Requirement already satisfied: pygments>=2.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (2.18.0)
    Requirement already satisfied: stack-data in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (0.6.2)
    Requirement already satisfied: typing-extensions>=4.6 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (4.12.2)
    Requirement already satisfied: pexpect>4.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipython>=6.1.0->ipywidgets) (4.9.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets) (0.8.4)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython>=6.1.0->ipywidgets) (0.2.13)
    Requirement already satisfied: executing>=1.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (0.2.3)
    Requirement already satisfied: six>=1.12.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets) (1.16.0)
    Using cached ipywidgets-8.1.3-py3-none-any.whl (139 kB)
    Using cached jupyterlab_widgets-3.0.11-py3-none-any.whl (214 kB)
    Using cached widgetsnbextension-4.0.11-py3-none-any.whl (2.3 MB)
    Installing collected packages: widgetsnbextension, jupyterlab-widgets, ipywidgets
    Successfully installed ipywidgets-8.1.3 jupyterlab-widgets-3.0.11 widgetsnbextension-4.0.11
    Note: you may need to restart the kernel to use updated packages.
    Collecting llama-index
      Using cached llama_index-0.10.59-py3-none-any.whl.metadata (11 kB)
    Collecting llama-index-agent-openai<0.3.0,>=0.1.4 (from llama-index)
      Using cached llama_index_agent_openai-0.2.9-py3-none-any.whl.metadata (729 bytes)
    Collecting llama-index-cli<0.2.0,>=0.1.2 (from llama-index)
      Using cached llama_index_cli-0.1.13-py3-none-any.whl.metadata (1.5 kB)
    Collecting llama-index-core==0.10.59 (from llama-index)
      Using cached llama_index_core-0.10.59-py3-none-any.whl.metadata (2.4 kB)
    Collecting llama-index-embeddings-openai<0.2.0,>=0.1.5 (from llama-index)
      Using cached llama_index_embeddings_openai-0.1.11-py3-none-any.whl.metadata (655 bytes)
    Collecting llama-index-indices-managed-llama-cloud>=0.2.0 (from llama-index)
      Using cached llama_index_indices_managed_llama_cloud-0.2.7-py3-none-any.whl.metadata (3.8 kB)
    Collecting llama-index-legacy<0.10.0,>=0.9.48 (from llama-index)
      Using cached llama_index_legacy-0.9.48-py3-none-any.whl.metadata (8.5 kB)
    Collecting llama-index-llms-openai<0.2.0,>=0.1.27 (from llama-index)
      Using cached llama_index_llms_openai-0.1.27-py3-none-any.whl.metadata (610 bytes)
    Collecting llama-index-multi-modal-llms-openai<0.2.0,>=0.1.3 (from llama-index)
      Using cached llama_index_multi_modal_llms_openai-0.1.8-py3-none-any.whl.metadata (728 bytes)
    Collecting llama-index-program-openai<0.2.0,>=0.1.3 (from llama-index)
      Using cached llama_index_program_openai-0.1.7-py3-none-any.whl.metadata (760 bytes)
    Collecting llama-index-question-gen-openai<0.2.0,>=0.1.2 (from llama-index)
      Using cached llama_index_question_gen_openai-0.1.3-py3-none-any.whl.metadata (785 bytes)
    Collecting llama-index-readers-file<0.2.0,>=0.1.4 (from llama-index)
      Using cached llama_index_readers_file-0.1.32-py3-none-any.whl.metadata (5.4 kB)
    Collecting llama-index-readers-llama-parse>=0.1.2 (from llama-index)
      Using cached llama_index_readers_llama_parse-0.1.6-py3-none-any.whl.metadata (3.6 kB)
    Collecting PyYAML>=6.0.1 (from llama-index-core==0.10.59->llama-index)
      Using cached PyYAML-6.0.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (2.1 kB)
    Collecting SQLAlchemy>=1.4.49 (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core==0.10.59->llama-index)
      Using cached SQLAlchemy-2.0.31-cp311-cp311-macosx_11_0_arm64.whl.metadata (9.6 kB)
    Collecting aiohttp<4.0.0,>=3.8.6 (from llama-index-core==0.10.59->llama-index)
      Using cached aiohttp-3.10.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (7.5 kB)
    Collecting dataclasses-json (from llama-index-core==0.10.59->llama-index)
      Using cached dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
    Collecting deprecated>=1.2.9.3 (from llama-index-core==0.10.59->llama-index)
      Using cached Deprecated-1.2.14-py2.py3-none-any.whl.metadata (5.4 kB)
    Collecting dirtyjson<2.0.0,>=1.0.8 (from llama-index-core==0.10.59->llama-index)
      Using cached dirtyjson-1.0.8-py3-none-any.whl.metadata (11 kB)
    Collecting fsspec>=2023.5.0 (from llama-index-core==0.10.59->llama-index)
      Using cached fsspec-2024.6.1-py3-none-any.whl.metadata (11 kB)
    Collecting httpx (from llama-index-core==0.10.59->llama-index)
      Using cached httpx-0.27.0-py3-none-any.whl.metadata (7.2 kB)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core==0.10.59->llama-index) (1.6.0)
    Collecting networkx>=3.0 (from llama-index-core==0.10.59->llama-index)
      Using cached networkx-3.3-py3-none-any.whl.metadata (5.1 kB)
    Collecting nltk<4.0.0,>=3.8.1 (from llama-index-core==0.10.59->llama-index)
      Using cached nltk-3.8.1-py3-none-any.whl.metadata (2.8 kB)
    Collecting numpy<2.0.0 (from llama-index-core==0.10.59->llama-index)
      Using cached numpy-1.26.4-cp311-cp311-macosx_11_0_arm64.whl.metadata (114 kB)
    Collecting openai>=1.1.0 (from llama-index-core==0.10.59->llama-index)
      Using cached openai-1.38.0-py3-none-any.whl.metadata (22 kB)
    Collecting pandas (from llama-index-core==0.10.59->llama-index)
      Using cached pandas-2.2.2-cp311-cp311-macosx_11_0_arm64.whl.metadata (19 kB)
    Collecting pillow>=9.0.0 (from llama-index-core==0.10.59->llama-index)
      Using cached pillow-10.4.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (9.2 kB)
    Collecting requests>=2.31.0 (from llama-index-core==0.10.59->llama-index)
      Using cached requests-2.32.3-py3-none-any.whl.metadata (4.6 kB)
    Collecting tenacity!=8.4.0,<9.0.0,>=8.2.0 (from llama-index-core==0.10.59->llama-index)
      Using cached tenacity-8.5.0-py3-none-any.whl.metadata (1.2 kB)
    Collecting tiktoken>=0.3.3 (from llama-index-core==0.10.59->llama-index)
      Using cached tiktoken-0.7.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.6 kB)
    Collecting tqdm<5.0.0,>=4.66.1 (from llama-index-core==0.10.59->llama-index)
      Using cached tqdm-4.66.4-py3-none-any.whl.metadata (57 kB)
    Requirement already satisfied: typing-extensions>=4.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core==0.10.59->llama-index) (4.12.2)
    Collecting typing-inspect>=0.8.0 (from llama-index-core==0.10.59->llama-index)
      Using cached typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)
    Collecting wrapt (from llama-index-core==0.10.59->llama-index)
      Using cached wrapt-1.16.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.6 kB)
    Collecting llama-cloud>=0.0.11 (from llama-index-indices-managed-llama-cloud>=0.2.0->llama-index)
      Using cached llama_cloud-0.0.11-py3-none-any.whl.metadata (751 bytes)
    Collecting beautifulsoup4<5.0.0,>=4.12.3 (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index)
      Using cached beautifulsoup4-4.12.3-py3-none-any.whl.metadata (3.8 kB)
    Collecting pypdf<5.0.0,>=4.0.1 (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index)
      Using cached pypdf-4.3.1-py3-none-any.whl.metadata (7.4 kB)
    Collecting striprtf<0.0.27,>=0.0.26 (from llama-index-readers-file<0.2.0,>=0.1.4->llama-index)
      Using cached striprtf-0.0.26-py3-none-any.whl.metadata (2.1 kB)
    Collecting llama-parse>=0.4.0 (from llama-index-readers-llama-parse>=0.1.2->llama-index)
      Using cached llama_parse-0.4.9-py3-none-any.whl.metadata (4.4 kB)
    Collecting aiohappyeyeballs>=2.3.0 (from aiohttp<4.0.0,>=3.8.6->llama-index-core==0.10.59->llama-index)
      Using cached aiohappyeyeballs-2.3.4-py3-none-any.whl.metadata (5.6 kB)
    Collecting aiosignal>=1.1.2 (from aiohttp<4.0.0,>=3.8.6->llama-index-core==0.10.59->llama-index)
      Using cached aiosignal-1.3.1-py3-none-any.whl.metadata (4.0 kB)
    Collecting attrs>=17.3.0 (from aiohttp<4.0.0,>=3.8.6->llama-index-core==0.10.59->llama-index)
      Using cached attrs-24.1.0-py3-none-any.whl.metadata (14 kB)
    Collecting frozenlist>=1.1.1 (from aiohttp<4.0.0,>=3.8.6->llama-index-core==0.10.59->llama-index)
      Using cached frozenlist-1.4.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (12 kB)
    Collecting multidict<7.0,>=4.5 (from aiohttp<4.0.0,>=3.8.6->llama-index-core==0.10.59->llama-index)
      Using cached multidict-6.0.5-cp311-cp311-macosx_11_0_arm64.whl.metadata (4.2 kB)
    Collecting yarl<2.0,>=1.0 (from aiohttp<4.0.0,>=3.8.6->llama-index-core==0.10.59->llama-index)
      Using cached yarl-1.9.4-cp311-cp311-macosx_11_0_arm64.whl.metadata (31 kB)
    Collecting soupsieve>1.2 (from beautifulsoup4<5.0.0,>=4.12.3->llama-index-readers-file<0.2.0,>=0.1.4->llama-index)
      Using cached soupsieve-2.5-py3-none-any.whl.metadata (4.7 kB)
    Collecting pydantic>=1.10 (from llama-cloud>=0.0.11->llama-index-indices-managed-llama-cloud>=0.2.0->llama-index)
      Using cached pydantic-2.8.2-py3-none-any.whl.metadata (125 kB)
    Collecting anyio (from httpx->llama-index-core==0.10.59->llama-index)
      Using cached anyio-4.4.0-py3-none-any.whl.metadata (4.6 kB)
    Collecting certifi (from httpx->llama-index-core==0.10.59->llama-index)
      Using cached certifi-2024.7.4-py3-none-any.whl.metadata (2.2 kB)
    Collecting httpcore==1.* (from httpx->llama-index-core==0.10.59->llama-index)
      Using cached httpcore-1.0.5-py3-none-any.whl.metadata (20 kB)
    Collecting idna (from httpx->llama-index-core==0.10.59->llama-index)
      Using cached idna-3.7-py3-none-any.whl.metadata (9.9 kB)
    Collecting sniffio (from httpx->llama-index-core==0.10.59->llama-index)
      Using cached sniffio-1.3.1-py3-none-any.whl.metadata (3.9 kB)
    Collecting h11<0.15,>=0.13 (from httpcore==1.*->httpx->llama-index-core==0.10.59->llama-index)
      Using cached h11-0.14.0-py3-none-any.whl.metadata (8.2 kB)
    Collecting click (from nltk<4.0.0,>=3.8.1->llama-index-core==0.10.59->llama-index)
      Using cached click-8.1.7-py3-none-any.whl.metadata (3.0 kB)
    Collecting joblib (from nltk<4.0.0,>=3.8.1->llama-index-core==0.10.59->llama-index)
      Using cached joblib-1.4.2-py3-none-any.whl.metadata (5.4 kB)
    Collecting regex>=2021.8.3 (from nltk<4.0.0,>=3.8.1->llama-index-core==0.10.59->llama-index)
      Using cached regex-2024.7.24-cp311-cp311-macosx_11_0_arm64.whl.metadata (40 kB)
    Collecting distro<2,>=1.7.0 (from openai>=1.1.0->llama-index-core==0.10.59->llama-index)
      Using cached distro-1.9.0-py3-none-any.whl.metadata (6.8 kB)
    Collecting charset-normalizer<4,>=2 (from requests>=2.31.0->llama-index-core==0.10.59->llama-index)
      Using cached charset_normalizer-3.3.2-cp311-cp311-macosx_11_0_arm64.whl.metadata (33 kB)
    Collecting urllib3<3,>=1.21.1 (from requests>=2.31.0->llama-index-core==0.10.59->llama-index)
      Using cached urllib3-2.2.2-py3-none-any.whl.metadata (6.4 kB)
    Collecting greenlet!=0.4.17 (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core==0.10.59->llama-index)
      Using cached greenlet-3.0.3-cp311-cp311-macosx_11_0_universal2.whl.metadata (3.8 kB)
    Collecting mypy-extensions>=0.3.0 (from typing-inspect>=0.8.0->llama-index-core==0.10.59->llama-index)
      Using cached mypy_extensions-1.0.0-py3-none-any.whl.metadata (1.1 kB)
    Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json->llama-index-core==0.10.59->llama-index)
      Using cached marshmallow-3.21.3-py3-none-any.whl.metadata (7.1 kB)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core==0.10.59->llama-index) (2.9.0)
    Collecting pytz>=2020.1 (from pandas->llama-index-core==0.10.59->llama-index)
      Using cached pytz-2024.1-py2.py3-none-any.whl.metadata (22 kB)
    Collecting tzdata>=2022.7 (from pandas->llama-index-core==0.10.59->llama-index)
      Using cached tzdata-2024.1-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: packaging>=17.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from marshmallow<4.0.0,>=3.18.0->dataclasses-json->llama-index-core==0.10.59->llama-index) (24.1)
    Collecting annotated-types>=0.4.0 (from pydantic>=1.10->llama-cloud>=0.0.11->llama-index-indices-managed-llama-cloud>=0.2.0->llama-index)
      Using cached annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
    Collecting pydantic-core==2.20.1 (from pydantic>=1.10->llama-cloud>=0.0.11->llama-index-indices-managed-llama-cloud>=0.2.0->llama-index)
      Using cached pydantic_core-2.20.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.6 kB)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-core==0.10.59->llama-index) (1.16.0)
    Using cached llama_index-0.10.59-py3-none-any.whl (6.8 kB)
    Using cached llama_index_core-0.10.59-py3-none-any.whl (15.5 MB)
    Using cached llama_index_agent_openai-0.2.9-py3-none-any.whl (13 kB)
    Using cached llama_index_cli-0.1.13-py3-none-any.whl (27 kB)
    Using cached llama_index_embeddings_openai-0.1.11-py3-none-any.whl (6.3 kB)
    Using cached llama_index_indices_managed_llama_cloud-0.2.7-py3-none-any.whl (9.5 kB)
    Using cached llama_index_legacy-0.9.48-py3-none-any.whl (2.0 MB)
    Using cached llama_index_llms_openai-0.1.27-py3-none-any.whl (11 kB)
    Using cached llama_index_multi_modal_llms_openai-0.1.8-py3-none-any.whl (5.9 kB)
    Using cached llama_index_program_openai-0.1.7-py3-none-any.whl (5.3 kB)
    Using cached llama_index_question_gen_openai-0.1.3-py3-none-any.whl (2.9 kB)
    Using cached llama_index_readers_file-0.1.32-py3-none-any.whl (38 kB)
    Using cached llama_index_readers_llama_parse-0.1.6-py3-none-any.whl (2.5 kB)
    Using cached aiohttp-3.10.0-cp311-cp311-macosx_11_0_arm64.whl (384 kB)
    Using cached beautifulsoup4-4.12.3-py3-none-any.whl (147 kB)
    Using cached Deprecated-1.2.14-py2.py3-none-any.whl (9.6 kB)
    Using cached dirtyjson-1.0.8-py3-none-any.whl (25 kB)
    Using cached fsspec-2024.6.1-py3-none-any.whl (177 kB)
    Using cached llama_cloud-0.0.11-py3-none-any.whl (154 kB)
    Using cached httpx-0.27.0-py3-none-any.whl (75 kB)
    Using cached httpcore-1.0.5-py3-none-any.whl (77 kB)
    Using cached llama_parse-0.4.9-py3-none-any.whl (9.4 kB)
    Using cached networkx-3.3-py3-none-any.whl (1.7 MB)
    Using cached nltk-3.8.1-py3-none-any.whl (1.5 MB)
    Using cached numpy-1.26.4-cp311-cp311-macosx_11_0_arm64.whl (14.0 MB)
    Using cached openai-1.38.0-py3-none-any.whl (335 kB)
    Using cached pillow-10.4.0-cp311-cp311-macosx_11_0_arm64.whl (3.4 MB)
    Using cached pypdf-4.3.1-py3-none-any.whl (295 kB)
    Using cached PyYAML-6.0.1-cp311-cp311-macosx_11_0_arm64.whl (167 kB)
    Using cached requests-2.32.3-py3-none-any.whl (64 kB)
    Using cached SQLAlchemy-2.0.31-cp311-cp311-macosx_11_0_arm64.whl (2.1 MB)
    Using cached striprtf-0.0.26-py3-none-any.whl (6.9 kB)
    Using cached tenacity-8.5.0-py3-none-any.whl (28 kB)
    Using cached tiktoken-0.7.0-cp311-cp311-macosx_11_0_arm64.whl (907 kB)
    Using cached tqdm-4.66.4-py3-none-any.whl (78 kB)
    Using cached typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
    Using cached wrapt-1.16.0-cp311-cp311-macosx_11_0_arm64.whl (38 kB)
    Using cached dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
    Using cached pandas-2.2.2-cp311-cp311-macosx_11_0_arm64.whl (11.3 MB)
    Using cached aiohappyeyeballs-2.3.4-py3-none-any.whl (12 kB)
    Using cached aiosignal-1.3.1-py3-none-any.whl (7.6 kB)
    Using cached anyio-4.4.0-py3-none-any.whl (86 kB)
    Using cached attrs-24.1.0-py3-none-any.whl (63 kB)
    Using cached certifi-2024.7.4-py3-none-any.whl (162 kB)
    Using cached charset_normalizer-3.3.2-cp311-cp311-macosx_11_0_arm64.whl (118 kB)
    Using cached distro-1.9.0-py3-none-any.whl (20 kB)
    Using cached frozenlist-1.4.1-cp311-cp311-macosx_11_0_arm64.whl (53 kB)
    Using cached greenlet-3.0.3-cp311-cp311-macosx_11_0_universal2.whl (271 kB)
    Using cached idna-3.7-py3-none-any.whl (66 kB)
    Using cached marshmallow-3.21.3-py3-none-any.whl (49 kB)
    Using cached multidict-6.0.5-cp311-cp311-macosx_11_0_arm64.whl (30 kB)
    Using cached mypy_extensions-1.0.0-py3-none-any.whl (4.7 kB)
    Using cached pydantic-2.8.2-py3-none-any.whl (423 kB)
    Using cached pydantic_core-2.20.1-cp311-cp311-macosx_11_0_arm64.whl (1.8 MB)
    Using cached pytz-2024.1-py2.py3-none-any.whl (505 kB)
    Using cached regex-2024.7.24-cp311-cp311-macosx_11_0_arm64.whl (278 kB)
    Using cached sniffio-1.3.1-py3-none-any.whl (10 kB)
    Using cached soupsieve-2.5-py3-none-any.whl (36 kB)
    Using cached tzdata-2024.1-py2.py3-none-any.whl (345 kB)
    Using cached urllib3-2.2.2-py3-none-any.whl (121 kB)
    Using cached yarl-1.9.4-cp311-cp311-macosx_11_0_arm64.whl (81 kB)
    Using cached click-8.1.7-py3-none-any.whl (97 kB)
    Using cached joblib-1.4.2-py3-none-any.whl (301 kB)
    Using cached annotated_types-0.7.0-py3-none-any.whl (13 kB)
    Using cached h11-0.14.0-py3-none-any.whl (58 kB)
    Installing collected packages: striprtf, pytz, dirtyjson, wrapt, urllib3, tzdata, tqdm, tenacity, SQLAlchemy, soupsieve, sniffio, regex, PyYAML, pypdf, pydantic-core, pillow, numpy, networkx, mypy-extensions, multidict, marshmallow, joblib, idna, h11, greenlet, fsspec, frozenlist, distro, click, charset-normalizer, certifi, attrs, annotated-types, aiohappyeyeballs, yarl, typing-inspect, requests, pydantic, pandas, nltk, httpcore, deprecated, beautifulsoup4, anyio, aiosignal, tiktoken, httpx, dataclasses-json, aiohttp, openai, llama-cloud, llama-index-legacy, llama-index-core, llama-parse, llama-index-readers-file, llama-index-llms-openai, llama-index-indices-managed-llama-cloud, llama-index-embeddings-openai, llama-index-readers-llama-parse, llama-index-multi-modal-llms-openai, llama-index-cli, llama-index-agent-openai, llama-index-program-openai, llama-index-question-gen-openai, llama-index
    Successfully installed PyYAML-6.0.1 SQLAlchemy-2.0.31 aiohappyeyeballs-2.3.4 aiohttp-3.10.0 aiosignal-1.3.1 annotated-types-0.7.0 anyio-4.4.0 attrs-24.1.0 beautifulsoup4-4.12.3 certifi-2024.7.4 charset-normalizer-3.3.2 click-8.1.7 dataclasses-json-0.6.7 deprecated-1.2.14 dirtyjson-1.0.8 distro-1.9.0 frozenlist-1.4.1 fsspec-2024.6.1 greenlet-3.0.3 h11-0.14.0 httpcore-1.0.5 httpx-0.27.0 idna-3.7 joblib-1.4.2 llama-cloud-0.0.11 llama-index-0.10.59 llama-index-agent-openai-0.2.9 llama-index-cli-0.1.13 llama-index-core-0.10.59 llama-index-embeddings-openai-0.1.11 llama-index-indices-managed-llama-cloud-0.2.7 llama-index-legacy-0.9.48 llama-index-llms-openai-0.1.27 llama-index-multi-modal-llms-openai-0.1.8 llama-index-program-openai-0.1.7 llama-index-question-gen-openai-0.1.3 llama-index-readers-file-0.1.32 llama-index-readers-llama-parse-0.1.6 llama-parse-0.4.9 marshmallow-3.21.3 multidict-6.0.5 mypy-extensions-1.0.0 networkx-3.3 nltk-3.8.1 numpy-1.26.4 openai-1.38.0 pandas-2.2.2 pillow-10.4.0 pydantic-2.8.2 pydantic-core-2.20.1 pypdf-4.3.1 pytz-2024.1 regex-2024.7.24 requests-2.32.3 sniffio-1.3.1 soupsieve-2.5 striprtf-0.0.26 tenacity-8.5.0 tiktoken-0.7.0 tqdm-4.66.4 typing-inspect-0.9.0 tzdata-2024.1 urllib3-2.2.2 wrapt-1.16.0 yarl-1.9.4
    Note: you may need to restart the kernel to use updated packages.
    Collecting llama-index-embeddings-huggingface
      Using cached llama_index_embeddings_huggingface-0.2.2-py3-none-any.whl.metadata (769 bytes)
    Collecting huggingface-hub>=0.19.0 (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface)
      Using cached huggingface_hub-0.24.5-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-embeddings-huggingface) (0.10.59)
    Collecting sentence-transformers>=2.6.1 (from llama-index-embeddings-huggingface)
      Using cached sentence_transformers-3.0.1-py3-none-any.whl.metadata (10 kB)
    Collecting filelock (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface)
      Using cached filelock-3.15.4-py3-none-any.whl.metadata (2.9 kB)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2024.6.1)
    Requirement already satisfied: packaging>=20.9 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (6.0.1)
    Requirement already satisfied: requests in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.32.3)
    Requirement already satisfied: tqdm>=4.42.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (4.66.4)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (4.12.2)
    Requirement already satisfied: aiohttp in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (3.10.0)
    Collecting minijinja>=1.0 (from huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface)
      Using cached minijinja-2.0.1-cp38-abi3-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl.metadata (8.8 kB)
    Requirement already satisfied: SQLAlchemy>=1.4.49 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.0.31)
    Requirement already satisfied: dataclasses-json in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.6.7)
    Requirement already satisfied: deprecated>=1.2.9.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.2.14)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.0.8)
    Requirement already satisfied: httpx in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.27.0)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.3)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.8.1)
    Requirement already satisfied: numpy<2.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.26.4)
    Requirement already satisfied: openai>=1.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.38.0)
    Requirement already satisfied: pandas in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.2.2)
    Requirement already satisfied: pillow>=9.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (10.4.0)
    Requirement already satisfied: tenacity!=8.4.0,<9.0.0,>=8.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (8.5.0)
    Requirement already satisfied: tiktoken>=0.3.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.7.0)
    Requirement already satisfied: typing-inspect>=0.8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.9.0)
    Requirement already satisfied: wrapt in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.16.0)
    Collecting transformers<5.0.0,>=4.34.0 (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached transformers-4.43.3-py3-none-any.whl.metadata (43 kB)
    Collecting torch>=1.11.0 (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached torch-2.4.0-cp311-none-macosx_11_0_arm64.whl.metadata (26 kB)
    Collecting scikit-learn (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached scikit_learn-1.5.1-cp311-cp311-macosx_12_0_arm64.whl.metadata (12 kB)
    Collecting scipy (from sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached scipy-1.14.0-cp311-cp311-macosx_14_0_arm64.whl.metadata (60 kB)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.3.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (24.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (1.9.4)
    Requirement already satisfied: click in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (8.1.7)
    Requirement already satisfied: joblib in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2024.7.24)
    Requirement already satisfied: anyio<5,>=3.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (4.4.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.9.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.8.2)
    Requirement already satisfied: sniffio in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.3.1)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.0.5)
    Requirement already satisfied: idna in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.7)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.14.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.0->huggingface-hub[inference]>=0.19.0->llama-index-embeddings-huggingface) (2.2.2)
    Requirement already satisfied: greenlet!=0.4.17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.0.3)
    Collecting sympy (from torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached sympy-1.13.1-py3-none-any.whl.metadata (12 kB)
    Collecting jinja2 (from torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached jinja2-3.1.4-py3-none-any.whl.metadata (2.6 kB)
    Collecting safetensors>=0.4.1 (from transformers<5.0.0,>=4.34.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached safetensors-0.4.3-cp311-cp311-macosx_11_0_arm64.whl.metadata (3.8 kB)
    Collecting tokenizers<0.20,>=0.19 (from transformers<5.0.0,>=4.34.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached tokenizers-0.19.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.7 kB)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (3.21.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.9.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2024.1)
    Collecting threadpoolctl>=3.1.0 (from scikit-learn->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached threadpoolctl-3.5.0-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (2.20.1)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-embeddings-huggingface) (1.16.0)
    Collecting MarkupSafe>=2.0 (from jinja2->torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached MarkupSafe-2.1.5-cp311-cp311-macosx_10_9_universal2.whl.metadata (3.0 kB)
    Collecting mpmath<1.4,>=1.1.0 (from sympy->torch>=1.11.0->sentence-transformers>=2.6.1->llama-index-embeddings-huggingface)
      Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
    Using cached llama_index_embeddings_huggingface-0.2.2-py3-none-any.whl (7.2 kB)
    Using cached huggingface_hub-0.24.5-py3-none-any.whl (417 kB)
    Using cached sentence_transformers-3.0.1-py3-none-any.whl (227 kB)
    Using cached minijinja-2.0.1-cp38-abi3-macosx_10_12_x86_64.macosx_11_0_arm64.macosx_10_12_universal2.whl (1.6 MB)
    Using cached torch-2.4.0-cp311-none-macosx_11_0_arm64.whl (62.1 MB)
    Using cached transformers-4.43.3-py3-none-any.whl (9.4 MB)
    Using cached filelock-3.15.4-py3-none-any.whl (16 kB)
    Using cached scikit_learn-1.5.1-cp311-cp311-macosx_12_0_arm64.whl (11.0 MB)
    Using cached scipy-1.14.0-cp311-cp311-macosx_14_0_arm64.whl (23.1 MB)
    Using cached safetensors-0.4.3-cp311-cp311-macosx_11_0_arm64.whl (410 kB)
    Using cached threadpoolctl-3.5.0-py3-none-any.whl (18 kB)
    Using cached tokenizers-0.19.1-cp311-cp311-macosx_11_0_arm64.whl (2.4 MB)
    Using cached jinja2-3.1.4-py3-none-any.whl (133 kB)
    Using cached sympy-1.13.1-py3-none-any.whl (6.2 MB)
    Using cached MarkupSafe-2.1.5-cp311-cp311-macosx_10_9_universal2.whl (18 kB)
    Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
    Installing collected packages: mpmath, threadpoolctl, sympy, scipy, safetensors, minijinja, MarkupSafe, filelock, scikit-learn, jinja2, huggingface-hub, torch, tokenizers, transformers, sentence-transformers, llama-index-embeddings-huggingface
    Successfully installed MarkupSafe-2.1.5 filelock-3.15.4 huggingface-hub-0.24.5 jinja2-3.1.4 llama-index-embeddings-huggingface-0.2.2 minijinja-2.0.1 mpmath-1.3.0 safetensors-0.4.3 scikit-learn-1.5.1 scipy-1.14.0 sentence-transformers-3.0.1 sympy-1.13.1 threadpoolctl-3.5.0 tokenizers-0.19.1 torch-2.4.0 transformers-4.43.3
    Note: you may need to restart the kernel to use updated packages.
    Collecting llama-index-llms-groq
      Using cached llama_index_llms_groq-0.1.4-py3-none-any.whl.metadata (2.2 kB)
    Requirement already satisfied: llama-index-core<0.11.0,>=0.10.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-llms-groq) (0.10.59)
    Collecting llama-index-llms-openai-like<0.2.0,>=0.1.3 (from llama-index-llms-groq)
      Using cached llama_index_llms_openai_like-0.1.3-py3-none-any.whl.metadata (753 bytes)
    Requirement already satisfied: PyYAML>=6.0.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (6.0.1)
    Requirement already satisfied: SQLAlchemy>=1.4.49 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.0.31)
    Requirement already satisfied: aiohttp<4.0.0,>=3.8.6 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.10.0)
    Requirement already satisfied: dataclasses-json in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.6.7)
    Requirement already satisfied: deprecated>=1.2.9.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.2.14)
    Requirement already satisfied: dirtyjson<2.0.0,>=1.0.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.0.8)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.6.1)
    Requirement already satisfied: httpx in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.27.0)
    Requirement already satisfied: nest-asyncio<2.0.0,>=1.5.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.6.0)
    Requirement already satisfied: networkx>=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.3)
    Requirement already satisfied: nltk<4.0.0,>=3.8.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.8.1)
    Requirement already satisfied: numpy<2.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.26.4)
    Requirement already satisfied: openai>=1.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.38.0)
    Requirement already satisfied: pandas in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.2.2)
    Requirement already satisfied: pillow>=9.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (10.4.0)
    Requirement already satisfied: requests>=2.31.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.32.3)
    Requirement already satisfied: tenacity!=8.4.0,<9.0.0,>=8.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (8.5.0)
    Requirement already satisfied: tiktoken>=0.3.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.7.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.66.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (4.66.4)
    Requirement already satisfied: typing-extensions>=4.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (4.12.2)
    Requirement already satisfied: typing-inspect>=0.8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.9.0)
    Requirement already satisfied: wrapt in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.16.0)
    Requirement already satisfied: llama-index-llms-openai<0.2.0,>=0.1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.1.27)
    Requirement already satisfied: transformers<5.0.0,>=4.37.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (4.43.3)
    Requirement already satisfied: aiohappyeyeballs>=2.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.3.4)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.3.1)
    Requirement already satisfied: attrs>=17.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (24.1.0)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.4.1)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (6.0.5)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from aiohttp<4.0.0,>=3.8.6->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.9.4)
    Requirement already satisfied: click in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (8.1.7)
    Requirement already satisfied: joblib in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.4.2)
    Requirement already satisfied: regex>=2021.8.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nltk<4.0.0,>=3.8.1->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.7.24)
    Requirement already satisfied: anyio<5,>=3.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (4.4.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.9.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.8.2)
    Requirement already satisfied: sniffio in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.3.1)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.0.5)
    Requirement already satisfied: idna in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.7)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpcore==1.*->httpx->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.14.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.3.2)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests>=2.31.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.2.2)
    Requirement already satisfied: greenlet!=0.4.17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from SQLAlchemy[asyncio]>=1.4.49->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.0.3)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (3.15.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.24.5)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (24.1)
    Requirement already satisfied: safetensors>=0.4.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.4.3)
    Requirement already satisfied: tokenizers<0.20,>=0.19 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers<5.0.0,>=4.37.0->llama-index-llms-openai-like<0.2.0,>=0.1.3->llama-index-llms-groq) (0.19.1)
    Requirement already satisfied: mypy-extensions>=0.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typing-inspect>=0.8.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.0.0)
    Requirement already satisfied: marshmallow<4.0.0,>=3.18.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from dataclasses-json->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (3.21.3)
    Requirement already satisfied: python-dateutil>=2.8.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.9.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2024.1)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->openai>=1.1.0->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (2.20.1)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from python-dateutil>=2.8.2->pandas->llama-index-core<0.11.0,>=0.10.1->llama-index-llms-groq) (1.16.0)
    Using cached llama_index_llms_groq-0.1.4-py3-none-any.whl (2.9 kB)
    Using cached llama_index_llms_openai_like-0.1.3-py3-none-any.whl (3.0 kB)
    Installing collected packages: llama-index-llms-openai-like, llama-index-llms-groq
    Successfully installed llama-index-llms-groq-0.1.4 llama-index-llms-openai-like-0.1.3
    Note: you may need to restart the kernel to use updated packages.
    Collecting groq
      Using cached groq-0.9.0-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: anyio<5,>=3.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq) (4.4.0)
    Requirement already satisfied: distro<2,>=1.7.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq) (1.9.0)
    Requirement already satisfied: httpx<1,>=0.23.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq) (0.27.0)
    Requirement already satisfied: pydantic<3,>=1.9.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq) (2.8.2)
    Requirement already satisfied: sniffio in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq) (1.3.1)
    Requirement already satisfied: typing-extensions<5,>=4.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from groq) (4.12.2)
    Requirement already satisfied: idna>=2.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5,>=3.5.0->groq) (3.7)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx<1,>=0.23.0->groq) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx<1,>=0.23.0->groq) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpcore==1.*->httpx<1,>=0.23.0->groq) (0.14.0)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->groq) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic<3,>=1.9.0->groq) (2.20.1)
    Using cached groq-0.9.0-py3-none-any.whl (103 kB)
    Installing collected packages: groq
    Successfully installed groq-0.9.0
    Note: you may need to restart the kernel to use updated packages.
    Collecting gradio
      Using cached gradio-4.40.0-py3-none-any.whl.metadata (15 kB)
    Collecting aiofiles<24.0,>=22.0 (from gradio)
      Using cached aiofiles-23.2.1-py3-none-any.whl.metadata (9.7 kB)
    Requirement already satisfied: anyio<5.0,>=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (4.4.0)
    Collecting fastapi (from gradio)
      Using cached fastapi-0.112.0-py3-none-any.whl.metadata (27 kB)
    Collecting ffmpy (from gradio)
      Using cached ffmpy-0.4.0-py3-none-any.whl.metadata (2.9 kB)
    Collecting gradio-client==1.2.0 (from gradio)
      Using cached gradio_client-1.2.0-py3-none-any.whl.metadata (7.1 kB)
    Requirement already satisfied: httpx>=0.24.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.27.0)
    Requirement already satisfied: huggingface-hub>=0.19.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.24.5)
    Collecting importlib-resources<7.0,>=1.3 (from gradio)
      Using cached importlib_resources-6.4.0-py3-none-any.whl.metadata (3.9 kB)
    Requirement already satisfied: jinja2<4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (3.1.4)
    Requirement already satisfied: markupsafe~=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.1.5)
    Collecting matplotlib~=3.0 (from gradio)
      Using cached matplotlib-3.9.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (11 kB)
    Requirement already satisfied: numpy<3.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (1.26.4)
    Collecting orjson~=3.0 (from gradio)
      Using cached orjson-3.10.6-cp311-cp311-macosx_10_15_x86_64.macosx_11_0_arm64.macosx_10_15_universal2.whl.metadata (50 kB)
    Requirement already satisfied: packaging in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (24.1)
    Requirement already satisfied: pandas<3.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.2.2)
    Requirement already satisfied: pillow<11.0,>=8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (10.4.0)
    Requirement already satisfied: pydantic>=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.8.2)
    Collecting pydub (from gradio)
      Using cached pydub-0.25.1-py2.py3-none-any.whl.metadata (1.4 kB)
    Collecting python-multipart>=0.0.9 (from gradio)
      Using cached python_multipart-0.0.9-py3-none-any.whl.metadata (2.5 kB)
    Requirement already satisfied: pyyaml<7.0,>=5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (6.0.1)
    Collecting ruff>=0.2.2 (from gradio)
      Using cached ruff-0.5.6-py3-none-macosx_11_0_arm64.whl.metadata (24 kB)
    Collecting semantic-version~=2.0 (from gradio)
      Using cached semantic_version-2.10.0-py2.py3-none-any.whl.metadata (9.7 kB)
    Collecting tomlkit==0.12.0 (from gradio)
      Using cached tomlkit-0.12.0-py3-none-any.whl.metadata (2.7 kB)
    Collecting typer<1.0,>=0.12 (from gradio)
      Using cached typer-0.12.3-py3-none-any.whl.metadata (15 kB)
    Requirement already satisfied: typing-extensions~=4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (4.12.2)
    Requirement already satisfied: urllib3~=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.2.2)
    Collecting uvicorn>=0.14.0 (from gradio)
      Using cached uvicorn-0.30.5-py3-none-any.whl.metadata (6.6 kB)
    Requirement already satisfied: fsspec in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio-client==1.2.0->gradio) (2024.6.1)
    Collecting websockets<13.0,>=10.0 (from gradio-client==1.2.0->gradio)
      Using cached websockets-12.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.6 kB)
    Requirement already satisfied: idna>=2.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5.0,>=3.0->gradio) (3.7)
    Requirement already satisfied: sniffio>=1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx>=0.24.1->gradio) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx>=0.24.1->gradio) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (3.15.4)
    Requirement already satisfied: requests in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (2.32.3)
    Requirement already satisfied: tqdm>=4.42.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (4.66.4)
    Collecting contourpy>=1.0.1 (from matplotlib~=3.0->gradio)
      Using cached contourpy-1.2.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (5.8 kB)
    Collecting cycler>=0.10 (from matplotlib~=3.0->gradio)
      Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting fonttools>=4.22.0 (from matplotlib~=3.0->gradio)
      Using cached fonttools-4.53.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (162 kB)
    Collecting kiwisolver>=1.3.1 (from matplotlib~=3.0->gradio)
      Using cached kiwisolver-1.4.5-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.4 kB)
    Collecting pyparsing>=2.3.1 (from matplotlib~=3.0->gradio)
      Using cached pyparsing-3.1.2-py3-none-any.whl.metadata (5.1 kB)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (2.9.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas<3.0,>=1.0->gradio) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas<3.0,>=1.0->gradio) (2024.1)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic>=2.0->gradio) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic>=2.0->gradio) (2.20.1)
    Requirement already satisfied: click>=8.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typer<1.0,>=0.12->gradio) (8.1.7)
    Collecting shellingham>=1.3.0 (from typer<1.0,>=0.12->gradio)
      Using cached shellingham-1.5.4-py2.py3-none-any.whl.metadata (3.5 kB)
    Collecting rich>=10.11.0 (from typer<1.0,>=0.12->gradio)
      Using cached rich-13.7.1-py3-none-any.whl.metadata (18 kB)
    Collecting starlette<0.38.0,>=0.37.2 (from fastapi->gradio)
      Using cached starlette-0.37.2-py3-none-any.whl.metadata (5.9 kB)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib~=3.0->gradio) (1.16.0)
    Collecting markdown-it-py>=2.2.0 (from rich>=10.11.0->typer<1.0,>=0.12->gradio)
      Using cached markdown_it_py-3.0.0-py3-none-any.whl.metadata (6.9 kB)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.18.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.3->gradio) (3.3.2)
    Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio)
      Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
    Using cached gradio-4.40.0-py3-none-any.whl (12.5 MB)
    Using cached gradio_client-1.2.0-py3-none-any.whl (318 kB)
    Using cached tomlkit-0.12.0-py3-none-any.whl (37 kB)
    Using cached aiofiles-23.2.1-py3-none-any.whl (15 kB)
    Using cached importlib_resources-6.4.0-py3-none-any.whl (38 kB)
    Using cached matplotlib-3.9.1-cp311-cp311-macosx_11_0_arm64.whl (7.8 MB)
    Using cached orjson-3.10.6-cp311-cp311-macosx_10_15_x86_64.macosx_11_0_arm64.macosx_10_15_universal2.whl (250 kB)
    Using cached python_multipart-0.0.9-py3-none-any.whl (22 kB)
    Using cached ruff-0.5.6-py3-none-macosx_11_0_arm64.whl (8.2 MB)
    Using cached semantic_version-2.10.0-py2.py3-none-any.whl (15 kB)
    Using cached typer-0.12.3-py3-none-any.whl (47 kB)
    Using cached uvicorn-0.30.5-py3-none-any.whl (62 kB)
    Using cached fastapi-0.112.0-py3-none-any.whl (93 kB)
    Using cached ffmpy-0.4.0-py3-none-any.whl (5.8 kB)
    Using cached pydub-0.25.1-py2.py3-none-any.whl (32 kB)
    Using cached contourpy-1.2.1-cp311-cp311-macosx_11_0_arm64.whl (245 kB)
    Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Using cached fonttools-4.53.1-cp311-cp311-macosx_11_0_arm64.whl (2.2 MB)
    Using cached kiwisolver-1.4.5-cp311-cp311-macosx_11_0_arm64.whl (66 kB)
    Using cached pyparsing-3.1.2-py3-none-any.whl (103 kB)
    Using cached rich-13.7.1-py3-none-any.whl (240 kB)
    Using cached shellingham-1.5.4-py2.py3-none-any.whl (9.8 kB)
    Using cached starlette-0.37.2-py3-none-any.whl (71 kB)
    Using cached websockets-12.0-cp311-cp311-macosx_11_0_arm64.whl (121 kB)
    Using cached markdown_it_py-3.0.0-py3-none-any.whl (87 kB)
    Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
    Installing collected packages: pydub, websockets, uvicorn, tomlkit, shellingham, semantic-version, ruff, python-multipart, pyparsing, orjson, mdurl, kiwisolver, importlib-resources, fonttools, ffmpy, cycler, contourpy, aiofiles, starlette, matplotlib, markdown-it-py, rich, gradio-client, fastapi, typer, gradio
    Successfully installed aiofiles-23.2.1 contourpy-1.2.1 cycler-0.12.1 fastapi-0.112.0 ffmpy-0.4.0 fonttools-4.53.1 gradio-4.40.0 gradio-client-1.2.0 importlib-resources-6.4.0 kiwisolver-1.4.5 markdown-it-py-3.0.0 matplotlib-3.9.1 mdurl-0.1.2 orjson-3.10.6 pydub-0.25.1 pyparsing-3.1.2 python-multipart-0.0.9 rich-13.7.1 ruff-0.5.6 semantic-version-2.10.0 shellingham-1.5.4 starlette-0.37.2 tomlkit-0.12.0 typer-0.12.3 uvicorn-0.30.5 websockets-12.0
    Note: you may need to restart the kernel to use updated packages.


### Import Libraries


```python
import os
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.groq import Groq
import gradio as gr
```

### !!! ACTION ITEM !!!
Visit https://console.groq.com/keys and set up an API Key then replace `<GROQ_API_KEY>` below with the newly generated key.


```python
os.environ["GROQ_API_KEY"] = "<GROQ_API_KEY>"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

### Disable Tokenizer Parallelism Globally


```python
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

### Data Ingestion


```python
reader = SimpleDirectoryReader(input_files=["files/reversing-for-everyone.pdf"])
documents = reader.load_data()
```

### Chunking


```python
text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
nodes = text_splitter.get_nodes_from_documents(documents)
```

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
Settings.embed_model = embed_model
Settings.llm = llm
```

### Create Vector Store Index


```python
# Debug VectorStoreIndex
print("VectorStoreIndex initialization")
vector_index = VectorStoreIndex.from_documents(
    documents, 
    show_progress=True, 
    node_parser=nodes
)
```

    VectorStoreIndex initialization



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
index = load_index_from_storage(storage_context)
```

### Define Query Engine


```python
query_engine = index.as_query_engine()
```

#### Feed in user query


```python
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
    fn=query_function,
    inputs=gr.Textbox(label="Query"),
    outputs=gr.Textbox(label="Response")
)

iface.launch()
```

    Running on local URL:  http://127.0.0.1:7861
    
    To create a public link, set `share=True` in `launch()`.

![image](https://github.com/mytechnotalent/rea/blob/main/example.png?raw=true)

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)
