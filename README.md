![image](https://github.com/mytechnotalent/rea/blob/main/Reverse%20Engineering%20Assistant.png?raw=true)

## FREE Reverse Engineering Self-Study Course [HERE](https://github.com/mytechnotalent/Reverse-Engineering-Tutorial)

<br>

# Reverse Engineering Assistant
Reverse Engineering Assistant utilizing RAG and an LLM.

Retrieval-Augmented Generation (RAG) is a powerful technique in natural language processing (NLP) that combines retrieval-based methods with generative models to produce more accurate and contextually relevant outputs. This approach was introduced in the 2020 paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" by Facebook AI Research (FAIR) .

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




```python
%conda create -n rea python=3.11 --y
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
    
    
    Note: you may need to restart the kernel to use updated packages.



```python
%pip install -U ipywidgets 
%pip install -U torch 
%pip install -U transformers
%pip install -U gradio 
%pip install -U fitz 
%pip install -U frontend 
%pip install -U tools 
%pip install -U pymupdf
```

    Requirement already satisfied: ipywidgets in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (8.1.3)
    Requirement already satisfied: comm>=0.1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (0.2.2)
    Requirement already satisfied: ipython>=6.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (8.26.0)
    Requirement already satisfied: traitlets>=4.3.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (5.14.3)
    Requirement already satisfied: widgetsnbextension~=4.0.11 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (4.0.11)
    Requirement already satisfied: jupyterlab-widgets~=3.0.11 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from ipywidgets) (3.0.11)
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
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: torch in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (2.4.0)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch) (3.15.4)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch) (1.13.1)
    Requirement already satisfied: networkx in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch) (3.3)
    Requirement already satisfied: jinja2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from torch) (2024.6.1)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from sympy->torch) (1.3.0)
    Note: you may need to restart the kernel to use updated packages.
    Collecting transformers
      Using cached transformers-4.43.3-py3-none-any.whl.metadata (43 kB)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (3.15.4)
    Requirement already satisfied: huggingface-hub<1.0,>=0.23.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (0.24.5)
    Requirement already satisfied: numpy>=1.17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (2.0.1)
    Requirement already satisfied: packaging>=20.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (24.1)
    Requirement already satisfied: pyyaml>=5.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (6.0.1)
    Collecting regex!=2019.12.17 (from transformers)
      Using cached regex-2024.7.24-cp311-cp311-macosx_11_0_arm64.whl.metadata (40 kB)
    Requirement already satisfied: requests in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (2.32.3)
    Collecting safetensors>=0.4.1 (from transformers)
      Using cached safetensors-0.4.3-cp311-cp311-macosx_11_0_arm64.whl.metadata (3.8 kB)
    Collecting tokenizers<0.20,>=0.19 (from transformers)
      Using cached tokenizers-0.19.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (6.7 kB)
    Requirement already satisfied: tqdm>=4.27 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from transformers) (4.66.4)
    Requirement already satisfied: fsspec>=2023.5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (2024.6.1)
    Requirement already satisfied: typing-extensions>=3.7.4.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub<1.0,>=0.23.2->transformers) (4.12.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->transformers) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->transformers) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->transformers) (2.2.2)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->transformers) (2024.7.4)
    Using cached transformers-4.43.3-py3-none-any.whl (9.4 MB)
    Using cached regex-2024.7.24-cp311-cp311-macosx_11_0_arm64.whl (278 kB)
    Using cached safetensors-0.4.3-cp311-cp311-macosx_11_0_arm64.whl (410 kB)
    Using cached tokenizers-0.19.1-cp311-cp311-macosx_11_0_arm64.whl (2.4 MB)
    Installing collected packages: safetensors, regex, tokenizers, transformers
    Successfully installed regex-2024.7.24 safetensors-0.4.3 tokenizers-0.19.1 transformers-4.43.3
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: gradio in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (4.40.0)
    Requirement already satisfied: aiofiles<24.0,>=22.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (23.2.1)
    Requirement already satisfied: anyio<5.0,>=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (4.4.0)
    Requirement already satisfied: fastapi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.112.0)
    Requirement already satisfied: ffmpy in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.4.0)
    Requirement already satisfied: gradio-client==1.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (1.2.0)
    Requirement already satisfied: httpx>=0.24.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.27.0)
    Requirement already satisfied: huggingface-hub>=0.19.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.24.5)
    Requirement already satisfied: importlib-resources<7.0,>=1.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (6.4.0)
    Requirement already satisfied: jinja2<4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (3.1.4)
    Requirement already satisfied: markupsafe~=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.1.5)
    Requirement already satisfied: matplotlib~=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (3.9.1)
    Requirement already satisfied: numpy<3.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.0.1)
    Requirement already satisfied: orjson~=3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (3.10.6)
    Requirement already satisfied: packaging in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (24.1)
    Requirement already satisfied: pandas<3.0,>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.2.2)
    Requirement already satisfied: pillow<11.0,>=8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (10.4.0)
    Requirement already satisfied: pydantic>=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.8.2)
    Requirement already satisfied: pydub in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.25.1)
    Requirement already satisfied: python-multipart>=0.0.9 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.0.9)
    Requirement already satisfied: pyyaml<7.0,>=5.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (6.0.1)
    Requirement already satisfied: ruff>=0.2.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.5.6)
    Requirement already satisfied: semantic-version~=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.10.0)
    Requirement already satisfied: tomlkit==0.12.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.12.0)
    Requirement already satisfied: typer<1.0,>=0.12 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.12.3)
    Requirement already satisfied: typing-extensions~=4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (4.12.2)
    Requirement already satisfied: urllib3~=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (2.2.2)
    Requirement already satisfied: uvicorn>=0.14.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio) (0.30.5)
    Requirement already satisfied: fsspec in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio-client==1.2.0->gradio) (2024.6.1)
    Requirement already satisfied: websockets<13.0,>=10.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from gradio-client==1.2.0->gradio) (12.0)
    Requirement already satisfied: idna>=2.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5.0,>=3.0->gradio) (3.7)
    Requirement already satisfied: sniffio>=1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5.0,>=3.0->gradio) (1.3.1)
    Requirement already satisfied: certifi in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx>=0.24.1->gradio) (2024.7.4)
    Requirement already satisfied: httpcore==1.* in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpx>=0.24.1->gradio) (1.0.5)
    Requirement already satisfied: h11<0.15,>=0.13 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httpcore==1.*->httpx>=0.24.1->gradio) (0.14.0)
    Requirement already satisfied: filelock in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (3.15.4)
    Requirement already satisfied: requests in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (2.32.3)
    Requirement already satisfied: tqdm>=4.42.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from huggingface-hub>=0.19.3->gradio) (4.66.4)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (1.2.1)
    Requirement already satisfied: cycler>=0.10 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (4.53.1)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (1.4.5)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from matplotlib~=3.0->gradio) (2.9.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas<3.0,>=1.0->gradio) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas<3.0,>=1.0->gradio) (2024.1)
    Requirement already satisfied: annotated-types>=0.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic>=2.0->gradio) (0.7.0)
    Requirement already satisfied: pydantic-core==2.20.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pydantic>=2.0->gradio) (2.20.1)
    Requirement already satisfied: click>=8.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typer<1.0,>=0.12->gradio) (8.1.7)
    Requirement already satisfied: shellingham>=1.3.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typer<1.0,>=0.12->gradio) (1.5.4)
    Requirement already satisfied: rich>=10.11.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from typer<1.0,>=0.12->gradio) (13.7.1)
    Requirement already satisfied: starlette<0.38.0,>=0.37.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fastapi->gradio) (0.37.2)
    Requirement already satisfied: six>=1.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib~=3.0->gradio) (1.16.0)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from rich>=10.11.0->typer<1.0,>=0.12->gradio) (2.18.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests->huggingface-hub>=0.19.3->gradio) (3.3.2)
    Requirement already satisfied: mdurl~=0.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0,>=0.12->gradio) (0.1.2)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: fitz in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.0.1.dev2)
    Requirement already satisfied: configobj in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (5.0.8)
    Requirement already satisfied: configparser in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (7.0.0)
    Requirement already satisfied: httplib2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (0.22.0)
    Requirement already satisfied: nibabel in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (5.2.1)
    Requirement already satisfied: nipype in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (1.8.6)
    Requirement already satisfied: numpy in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (2.0.1)
    Requirement already satisfied: pandas in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (2.2.2)
    Requirement already satisfied: pyxnat in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (1.6.2)
    Requirement already satisfied: scipy in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from fitz) (1.14.0)
    Requirement already satisfied: six in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from configobj->fitz) (1.16.0)
    Requirement already satisfied: pyparsing!=3.0.0,!=3.0.1,!=3.0.2,!=3.0.3,<4,>=2.4.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from httplib2->fitz) (3.1.2)
    Requirement already satisfied: packaging>=17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nibabel->fitz) (24.1)
    Requirement already satisfied: click>=6.6.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (8.1.7)
    Requirement already satisfied: networkx>=2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (3.3)
    Requirement already satisfied: prov>=1.5.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (2.0.1)
    Requirement already satisfied: pydot>=1.2.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (3.0.1)
    Requirement already satisfied: python-dateutil>=2.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (2.9.0)
    Requirement already satisfied: rdflib>=5.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (6.3.2)
    Requirement already satisfied: simplejson>=3.8.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (3.19.2)
    Requirement already satisfied: traits!=5.0,<6.4,>=4.6 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (6.3.2)
    Requirement already satisfied: filelock>=3.0.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (3.15.4)
    Requirement already satisfied: etelemetry>=0.2.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (0.3.1)
    Requirement already satisfied: looseversion in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from nipype->fitz) (1.3.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->fitz) (2024.1)
    Requirement already satisfied: tzdata>=2022.7 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pandas->fitz) (2024.1)
    Requirement already satisfied: lxml>=4.3 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pyxnat->fitz) (5.2.2)
    Requirement already satisfied: requests>=2.20 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pyxnat->fitz) (2.32.3)
    Requirement already satisfied: pathlib>=1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pyxnat->fitz) (1.0.1)
    Requirement already satisfied: ci-info>=0.2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from etelemetry>=0.2.0->nipype->fitz) (0.3.0)
    Requirement already satisfied: isodate<0.7.0,>=0.6.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from rdflib>=5.0.0->nipype->fitz) (0.6.1)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests>=2.20->pyxnat->fitz) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests>=2.20->pyxnat->fitz) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests>=2.20->pyxnat->fitz) (2.2.2)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from requests>=2.20->pyxnat->fitz) (2024.7.4)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: frontend in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.0.3)
    Requirement already satisfied: starlette>=0.12.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from frontend) (0.37.2)
    Requirement already satisfied: uvicorn>=0.7.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from frontend) (0.30.5)
    Requirement already satisfied: itsdangerous>=1.1.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from frontend) (2.2.0)
    Requirement already satisfied: aiofiles in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from frontend) (23.2.1)
    Requirement already satisfied: anyio<5,>=3.4.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from starlette>=0.12.0->frontend) (4.4.0)
    Requirement already satisfied: click>=7.0 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from uvicorn>=0.7.1->frontend) (8.1.7)
    Requirement already satisfied: h11>=0.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from uvicorn>=0.7.1->frontend) (0.14.0)
    Requirement already satisfied: idna>=2.8 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5,>=3.4.0->starlette>=0.12.0->frontend) (3.7)
    Requirement already satisfied: sniffio>=1.1 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from anyio<5,>=3.4.0->starlette>=0.12.0->frontend) (1.3.1)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: tools in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (0.1.9)
    Requirement already satisfied: pytils in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from tools) (0.4.1)
    Requirement already satisfied: six in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from tools) (1.16.0)
    Requirement already satisfied: lxml in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from tools) (5.2.2)
    Note: you may need to restart the kernel to use updated packages.
    Requirement already satisfied: pymupdf in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (1.24.9)
    Requirement already satisfied: PyMuPDFb==1.24.9 in /opt/anaconda3/envs/rea/lib/python3.11/site-packages (from pymupdf) (1.24.9)
    Note: you may need to restart the kernel to use updated packages.



```python
import os
import torch
import fitz
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gradio as gr

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


class RAG:
    """
    A class for implementing the Retrieval-Augmented Generation (RAG) model with PDF
    document retrieval.
    """
    def __init__(self, model_name='t5-base'):
        """
        Initialize the RAG class with a specified model and tokenizer.

        Params:
            model_name: object
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)

    def read_pdf(self, file_path):
        """
        Read and extract text from a PDF file.

        Parmas:
            file_path: str
        """
        doc = fitz.open(file_path)
        text = ''
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text('text')  # Ensure to use the 'text' option for consistent extraction
        return text

    def clean_text(self, text):
        """
        Clean and preprocess extracted text to remove unwanted characters.

        Params:
            text: str
        """
        text = text.replace('\x0c', '')  # Remove form feed characters
        text = ' '.join(text.split())  # Replace multiple spaces with a single space
        return text

    def chunk_text(self, text, chunk_size=1024):
        """
        Split text into chunks of a specified size.

        Params:
            text: str
            chunk_size: int, optional
        """
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ''
        for paragraph in paragraphs:
            if len(self.tokenizer.encode(current_chunk + paragraph, add_special_tokens=False)) < chunk_size:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def generate_answer(self, query, context):
        """
        Generate an answer for a given query and context using the model.

        params:
            query: str
            context: str

        Returns:
            str
        """
        input_text = f'question: {query} context: {context}'
        inputs = self.tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=9,  # Use beam search for better results
            early_stopping=True
        )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

    def rag(self, query, pdf_dir, chunk_size=1024):
        """
        Perform Retrieval-Augmented Generation (RAG) on all PDFs in a directory with a given query.
        
        Params:
            query: str
            pdf_dir: str
            chunk_size: int, optional
        """
        pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
        answers = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_file)
            pdf_text = self.read_pdf(pdf_path)
            cleaned_text = self.clean_text(pdf_text)
            chunks = self.chunk_text(cleaned_text, chunk_size)
            for chunk in chunks:
                answer = self.generate_answer(query, chunk)
                answers.append(answer)
        return ' '.join(answers)


def rag_interface(query):
    """
    Interface function for Gradio to process the query with RAG.

    Params:
        query: str
    """
    rag = RAG()
    return rag.rag(query, pdf_dir='files')


if __name__ == '__main__':
    iface = gr.Interface(
        fn=rag_interface,
        inputs=gr.Textbox(label='Query', placeholder='Enter your query here...'),
        outputs=gr.Textbox(label='Answer'),
        live=False,  # Set to False to ensure that the submit button is shown
        title='Reverse Engineering Assistant',
        description='Enter a query to retrieve information from all PDFs in the files directory.'
    )
    iface.launch()

```

    Running on local URL:  http://127.0.0.1:7875
    
    To create a public link, set `share=True` in `launch()`.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0)

