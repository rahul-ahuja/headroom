## Background

Headroom is an open-source context compression layer for AI agents to reduce input tokens to be prcocessed by LLMs. At high-level the workflow is that there's a ContentRouter that detects content type and select the right compressor. In this report, Kompress-v2-base has been used to compress strings and document files. 

I have understood the main functionalities of the codebase and installed the headroom as a package on my local computer and also ran experiments on Kaggle and Google Colab. 

In this report, I have focused on improving the headroom's retrieval integration to langchain. According to the headroom's langchain [docs](https://headroomlabs-ai.github.io/headroom/langchain/), token savings should occur when invoking the model with the message "Hello!". However, the docs use OpenAI's GPT-4o as the example model. As per the email from hiring team member, I am not expected to use paid LLM vendors and use open-source LLMs instead such as HuggingFace, Nvidia and OpenRouter. As a result, I wasn't able to run the exact code from the docs to confirm the token savings or reproduce the issue using the example code given in the docs. Instead I ran their examples with HuggingFace API. I managed to run inference via the DeepSeek API locally, and separately downloaded and ran the Qwen LLM model on Kaggle (`notebooks/qwen-demo-kaggle.ipynb`). But I failed to compress the number of tokens using headroom API wrapped on the HuggingFace library. I also tried passing a longer context to Headroom's API, but was not able to compress and save tokens. I wasn't able to achieve any token savings with headroom's Chat Agent for text generation. I also encountered various issues in using headroom package.

I also encountered an error stating that `get_metrics` is not included in the package. 
`AttributeError: 'HeadroomChatModel' object has no attribute 'get_metrics'`

To confirm the correct package version is aligned with the codes given in the docs, I ran pip show and found Headroom's 0.37 version being used matches the current version referenced in the code examples given in the documentations.

This has been raised as an issue on the GitHub repo. [LINK]

## Introduction

For this report, I have extended `HeadroomDocumentCompressor` with additional techniques to reduce the number of retrieved documents. I adapted Example 2 ("RAG Pipeline with Document Filtering") from the [docs](https://headroomlabs-ai.github.io/headroom/langchain/), but encountered the issue because the `ContextualCompressionRetriever` has been moved to `langchain_classic`, meaning it is deprecated and no longer actively maintained. Therefore, it is NOT recommended to be used.

`HeadroomDocumentCompressor` is effectively a reranking technique. It selects a smaller, more relevant subset of documents from a larger retrieved set. The idea behind reranking is that the initial retrieval pass is broad and computationally cheap, pulling in many candidate documents, after which a second, more selective step narrows that set down to what the LLM actually uses. This compression step is conceptually similar to retrieval itself, it is just applied downstream.

In Headroom's `ContextualCompressionRetriever`, a naive BM25 algorithm is used for the selection of retrived documents. Because the compression algorithm operates on an already retrieved, smaller set of documents, it can afford to be more robust without becoming computationally expensive. For that reason, I decided to extend `ContextualCompressionRetriever` with multiple techniques such as BM25 with TF-IDF and with a transformer-based semantic search steps. Udacity course project [1] has been extensively used in this task.

## Dataset

**Natural Questions** dataset is used which consists of real Google search queries with Wikipedia answers.
This dataset provides a realistic evaluation of how different retrieval methods handle real-world information needs. More details about the data set can be found here https://pypi.org/project/beir/0.1.1/

## Methods / Approaches

Looking at the [source](`headroom/integrations/langchain/retriever.py`), `HeadroomDocumentCompressor`'s relevance scoring is a simplified BM25: it tokenizes the query and document content, applies the standard BM25 term-frequency formula with `k1=1.5` and `b=0.75`. To understand `HeadroomDocumentCompressor` implementation, we need to first understand BM25 retrieval algorithm. BM25 is an improved version of TF-IDF that introduces two additional parameters: term frequency saturation and document length normalization. BM25 measures the relevance of a document to a given query.
TF (term frequency) captures how often a word occurs within a document. The more frequently a word appears, the higher its TF score. IDF (inverse document frequency) scores a word based on how rare or distinctive it is across the entire corpus of documents, giving more weight to terms that are less common therefore more informative and less weight to terms that appear in most documents. BM25 improves on plain TF-IDF by adding saturation and normalization. Term frequency saturation ensures that a word's contribution to the score grows more slowly as it appears more times, preventing common or repeated words from dominating the score. Document length normalization ensures that longer documents aren't unfairly favored just because they contain more terms overall, some of which may not be relevant to the query.

`HeadroomDocumentCompressor` substitutes with a hardcoded `avg_dl=100` (average document length) in place of a real corpus derived average document length, and omits IDF effectively ignore the corpus statistics. By default, this BM25 score alone determines which documents are `max_documents`. Jaccard similarity is applied when `prefer_diverse=True`. After the top-scoring document is selected, each subsequent pick is chosen to maximize relevance while minimizing Jaccard similarity (over token sets) to documents already selected, discouraging near duplicate results rather than serving as a second relevance pass.

**The Challenge with Keywords:**
- Exact word matching only
- No understanding of synonyms or context
- Struggles with semantic similarity

In this report, I have extended The `HeadroomDocumentCompressor` with the `rank_bm25` package. Here are the scripts  (`src/bm25_retriever.py`) (`src/headroom_bm25.py`) and transformer based compressor (`src/transformer_retriever.py`) (`src/headroom_transformer.py`)

In the report, I have carried out the analysis and evaluated each techniques of naive BM25, standard BM25 and transformers models of miniLM, DistilBert, & BGE-small. I have also provided scripts to fine tune transformers to the corpus. Finetuning shows that we can improve the performance of evaluation results further. Transformers give more scope with customization to improve the performance. Transformers and BM25 learn the representation of the corpus by `build_index` method and then perform compression. Indexing is done only once so it is a sunk cost. We can then query from the index fast.

Transformers are the neural network architecture built around a self-attention mechanism that weighs the relevance of every token to every other token, regardless of distance that powers most modern language models. Transformers generate the embeddings from text tokens. Embeddings are numerical vector representations of data (words, sentences, images) where items with similar meaning end up close together in a high-dimensional space, typically learned by a neural network during training. We can use Cosine similarity to numerically score similarity between two vector embeddings. Cosine Similarity measures the angle between vectors ranging from -1 to 1 rather than their magnitude, so a score near 1 means the vectors point in nearly the same direction and represent similar content. Cosine similarity is used to compare those embeddings for tasks like semantic search, recommendations, or retrieval-augmented generation (RAG).

Transformer retrieval approach enables customizations. This notebook (`notebooks/finetuning-transformer-retrieval.ipynb`) fine-tunes a small sentence-transformer embedding model for dense retrieval and evaluates the improvement. Contrastive loss is used finetune the model with training pairs of (query, correct documents).

**The Power of Semantic Understanding:**
- Contextual word embeddings
- Understanding of synonyms and paraphrases
- Captures semantic meaning beyond exact words

**How Transformer Retrieval Works:**
1. **Processing**: Parallel processing, global attention, rich context
2. **Encode**: Transform text into dense vector representations
3. **Compare**: Use cosine similarity between query and document vectors
4. **Retrieve**: Find documents with highest semantic similarity

This notebook (`notebooks/semantic-transformers.ipynb`) focuses purely on the transformer/semantic side of retrieval, comparing several sentence-embedding models with the given information retrieval metrics through comparison tables and visualization. Comparison is among MiniLM-L6-v2 , MPNet-base, and BGE-small.

Model Architecture Comparison

MiniLM-L6-v2 (Small & Fast)
Main feature: Lightweight model, good efficiency
Size: 23M parameters, 6 layers
Training: Knowledge distillation from BERT
Strength: Speed and efficiency
Use Case: Real-time applications, resource-constrained environments

BERT-base/DistilBERT (Balanced)
Main feature: Classic transformer, balanced performance
Size: 67M parameters, 12 layers (6 for DistilBERT)
Training: Masked language modeling
Strength: Reliable general-purpose performance
Use Case: Standard NLP tasks requiring proven architecture

BGE-small (Retrieval-Optimized)
Main feature: Specifically trained for retrieval tasks
Size: 33M parameters
Training: Contrastive learning on query-document pairs
Strength: Optimized for retrieval and similarity
Use Case: Search and semantic retrieval tasks

Key Takeaway of model architectures
Transformer architectures make different trade-offs between size, speed, and task-specific performance. Smaller models can match or exceed larger ones when optimized for specific tasks.

## Codes of using the extension

{LINK} Below is the adapted version of Example 2 from Headroom Langchain [docs](https://headroomlabs-ai.github.io/headroom/langchain/) using the extensions developed from this report. These scripts of the extension of Headroom's Compressor has been suggested to the maintainer of headroom as an open-source contributions. In `notebooks/unified_retrieval_comparison`, BM25 algorithm does not retrieve correct document for the given query. Headroom's BM25 and Transformers model retrieves correct document. 

`query = "Which Doctor performed Caesarean?"` 

```
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import ContextualCompressionRetriever
from headroom_bm25 import HeadroomBM25DocumentCompressor
from headroom_transformer import HeadroomTransformerDocumentCompressor
from utils import measure_latency

# initial setup of embeddings and vector databases

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(documents, embeddings)

base_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 100})

query = "Which Doctor performed Caesarean?"
docs = base_retriever.invoke(query)

# Headroom's original compressor

headroom_compressor = HeadroomDocumentCompressor(
    max_documents=20,
    min_relevance=0.0,
    prefer_diverse=False,
)

@measure_latency
def headroom_compress(query: str):
    compressed = headroom_compressor.compress_documents(docs, query)
    return compressed

# usage
results = headroom_compress(query)
print(results[0].page_content)

#  ---> Retrieves Correct Output Document within 0.0032 seconds


# Original BM25 class extended by Headroom compressor class

#sunk cost of indexing
headroom_bm25 = HeadroomBM25DocumentCompressor(
    corpus_texts=corpus_texts, 
    max_documents=20,
    min_relevance=0.3,
)

@measure_latency
def headroom_bm25_compress(query: str):
    return headroom_bm25.compress_documents(docs, query)

# usage
results = headroom_bm25_compress("Which Doctor performed Caesarean?")
print(results[0].page_content)

#  ---> Retrieves Incorrect Output Document within 0.0247 seconds

# Transformer class extended by Headroom compressor class
#sunk cost of indexing
headroom_transformer = HeadroomTransformerDocumentCompressor(
    corpus_texts=corpus_texts, 
    max_documents=10,
    min_relevance=0.0,
)

@measure_latency
def headroom_transformer_compress(query: str):
    return headroom_transformer.compress_documents(docs, query)

# usage
results = headroom_transformer_compress("Which Doctor performed Caesarean?")
print(results[0].page_content)

#  ---> Retrieves Correct Output Document within 0.3311 seconds
```

### **🏗️ RAG Pipeline - Use case**
- **Retrieval**: Find relevant context documents
- **Augmentation**: Combine query + retrieved context  
- **Generation**: Use LLM to generate informed answers

### **Description of codes given here**

#### **🔧 `src/bm25_retriever.py` - Traditional Keyword Search**
- **Text tokenization** and preprocessing for BM25 scoring
- **BM25 index creation** using the rank-bm25 library
- **Retrieval logic** to find and rank top-k documents
- **Understanding**: Learn tf-idf concepts and traditional IR methods

**Key concepts:** Tokenization, BM25 scoring, keyword matching, document frequency

**Key concepts:** Word embeddings, vector averaging, cosine similarity, hyperparameter tuning

#### **🤖 `src/transformer_retriever.py` - Modern Semantic Search**
- **Corpus encoding** using sentence transformers
- **Query encoding** with the same transformer model
- **Semantic similarity computation** and top-k retrieval
- **Understanding**: Experience state-of-the-art semantic search

#### **🤖 `src/headroom_transformer.py` and `src/headroom_bm25.py`  - Extends HeadroomDocumentCompressor baseclass to HeadroomBM25DocumentCompressor and HeadroomTransformerDocumentCompressor **

**Key concepts:** Sentence embeddings, contextual understanding, transformer models, semantic similarity

#### **📁 `src/data_loader.py`**
This file handles the complex BeIR dataset loading and preprocessing.

#### **🛠️ `src/utils.py`**
Contains utility functions for text processing and system operations.

In the experiments a FAISS vector store retriever is built and benchmarks the latency and output of Headroom's compressor variants (base class, BM25-based, and transformer-based document compression) as a `ContextualCompressionRetriever` wrapped around that base retriever, essentially demonstrating how Headroom's compression layer can be used to filter/re-rank an initial large candidate set. An example, top-100 FAISS results down to a smaller, more relevant set while measuring the added latency.

## 📊 Below is explanation of the Evaluation Metrics

**Recall@k**: Found relevant docs / Total relevant docs  
*Example: 3 found out of 5 relevant → Recall@5 = 60%*  
**Why**: Measures completeness - did we miss important information?

**Precision@k**: Relevant docs in top-k / k  
*Example: 3 relevant out of 5 returned → Precision@5 = 60%*  
**Why**: Measures quality - are results actually useful?

**MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant doc  
*Example: First relevant at position 2 → RR = 0.5*  
**Why**: Measures efficiency - how fast do users find answers?

**k values**: k=1 (mobile), k=5 (above fold), k=10 (max users check)

- **Recall@k calculation**: Fraction of relevant documents found
- **Precision@k calculation**: Fraction of retrieved documents that are relevant  
- **Mean Reciprocal Rank (MRR)**: Quality of first relevant result
- **Understanding**: Learn how to measure retrieval system performance

### **📈 Evaluation Metrics**
- **Recall@k**: What fraction of relevant documents are retrieved?
- **Precision@k**: What fraction of retrieved documents are relevant?
- **MRR (Mean Reciprocal Rank)**: How quickly do we find the first relevant document?

There is a tradeoff of computational time in using transformer compressor with accuracy performance boost.

### **Quality Assurance: Unit Tests** ✅
Unit testing has been carried out to ensure code robustness, Your implementation is validated by a comprehensive test suite:

- **Core Functionality**: Indexing, retrieval, ranking for all methods
- **Semantic Understanding**: Synonym handling, context awareness
- **Edge Cases**: Empty queries, out-of-vocabulary words
- **Consistency**: Deterministic behavior, proper parameter handling
- **Integration**: Component interaction and data flow


Maintainer seems to be actively working on Langchain fixes as seen in the commit https://github.com/headroomlabs-ai/headroom/commit/5d025f7a03870a402918e6425ca9fcd40400edb2

Conclusion: Given the issues highlighted in my analyzes I would be hasistant to use headroom package for production systems or as a third party library. This package seems to be still on work in progress and not well configured by looking at the commits as well. 

Reference:

[1] Udacity project course https://github.com/udacity/cd1822-seq-models-transformers-public/tree/main/project/starter
