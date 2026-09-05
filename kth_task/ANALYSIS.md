NOTES:
Have a demo folder
Explain evaluations
Learn and explain BM25 
Remove TODOs from Udacity scripts



Headroom is an open-source context-compression layer for AI agents to reduce input tokens to be prcocessed by LLMs. At high-level the workflow is that there's a ContentRouter that detects content type and select the right compressor. In this ask, Kompress-v2-base has been used to compress strings and document files. 

I have read and setup the headroom as a package on my local computer and also ran experiments on Kaggle and Google Colab. 

I ran the Headroom agent but wasn't able to achieve any token savings. I also encountered various issues in using Headroom package.  


According to the headroom's langchain docs(https://headroomlabs-ai.github.io/headroom/langchain/), token savings should occur when invoking the model with the message "Hello!". However, the docs use OpenAI's GPT-4o as the example model. As per the email guidance, I was expected not to use paid LLM vendors and to use open-source LLMs instead such as HuggingFace, Nvidia and OpenRouter. As a result, I wasn't able to run the exact code from the docs to confirm the token savings or reproduce the issue using the example code given in the docs. Instead I ran their examples with HuggingFace API. I managed to run inference via the DeepSeek API locally, and separately downloaded and ran the Qwen LLM model on Kaggle. [file location]. But I failed to compress the tokens using headroom API wrapped on the HuggingFace library. I also tried passing a longer context to Headroom's API, but was not able to compress and save tokens.


I also encountered an error stating that get_metrics is not included in the package. 
`AttributeError: 'HeadroomChatModel' object has no attribute 'get_metrics'`

To confirm I was using the correct package version, I ran pip show and found I'm on Headroom 0.37, which matches the current version referenced in the documentation I've been using for the code examples.

This has been raised as an issue on the GitHub repo. [LINK]


For this report, I have extended `HeadroomDocumentCompressor` with additional techniques to reduce the number of retrieved documents. I adapted Example 2 ("RAG Pipeline with Document Filtering") from the docs, but ran into the fact that `ContextualCompressionRetriever` has been moved to `langchain_classic`, meaning it is deprecated and no longer actively maintained yet it is still the class Headroom relies on.

`HeadroomDocumentCompressor` is effectively a reranking technique: it selects a smaller, more relevant subset of documents from a larger retrieved set. The idea behind reranking is that the initial retrieval pass is broad and computationally cheap, pulling in many candidate documents, after which a second, more selective step narrows that set down to what the LLM actually uses. This compression step is conceptually similar to retrieval itself, just applied downstream.

In Headroom's `ContextualCompressionRetriever`, a naive BM25 algorithm is used for this selection. Because the compression algorithm operates on an already-retrieved, smaller set of documents, it can afford to be more robust without becoming computationally expensive. For that reason, I decided to extend `ContextualCompressionRetriever` with multiple techniques such as BM25 with TF-IDF (confirm??) and with a transformer-based semantic search step.

Looking at the source directly (`headroom/integrations/langchain/retriever.py`), `HeadroomDocumentCompressor`'s relevance scoring is a simplified BM25: it tokenizes the query and document content, applies the standard BM25 term-frequency formula with `k1=1.5` and `b=0.75`. To understand `HeadroomDocumentCompressor` implementation, we need to first understand BM25 retrieval algorithm. BM25 is an improved version of TF-IDF that introduces two additional parameters: term frequency saturation and document length normalization. BM25 measures the relevance of a document to a given query.
TF (term frequency) captures how often a word occurs within a document. The more frequently a word appears, the higher its TF score. IDF (inverse document frequency) scores a word based on how rare or distinctive it is across the entire corpus of documents, giving more weight to terms that are less common therefore more informative and less weight to terms that appear in most documents.

BM25 improves on plain TF-IDF by adding saturation and normalization. Term frequency saturation ensures that a word's contribution to the score grows more slowly as it appears more times, preventing common or repeated words from dominating the score. Document length normalization ensures that longer documents aren't unfairly favored just because they contain more terms overall, some of which may not be relevant to the query.

`HeadroomDocumentCompressor` substitutes a hardcoded `avg_dl=100` in place of a real corpus-derived average document length, and omits IDF effectively ignore the corpus statistics. By default, this BM25 score alone determines which documents are `max_documents`. Jaccard similarity is applied when `prefer_diverse=True`, which selects an MMR-style approach. After the top-scoring document is selected, each subsequent pick is chosen to maximize relevance while minimizing Jaccard similarity (over token sets) to documents already selected, discouraging near-duplicate results rather than serving as a second relevance pass. (Clarify the content)

**The Challenge with Keywords:**
- Exact word matching only
- No understanding of synonyms or context
- Struggles with semantic similarity

In this report, I have extended The `HeadroomDocumentCompressor` with the `rank_bm25` package: https://github.com/rahul-ahuja/cd1822-seq-models-transformers-public/blob/main/project/starter/src/bm25_retriever.py 


In my implementation I have carried out the analysis and evaluated each ot the techniques of naive BM25, standard BM25 and transformers models of miniLLM, DistilBert, & _model_placeholder_. I have also provided scripts to finetune transfotmers to the corpus. Finetuning evaluation results shows that we can improve the performance further. Transformers gives more scope to improve the performance. 

These techniques learn the representation of the corpus by build_index method and then perform compression (retrieval). Indexing is done only once so it is a sunk cost. Then we can query from the index fast.


**The Power of Semantic Understanding:**
- Contextual word embeddings
- Understanding of synonyms and paraphrases
- Captures semantic meaning beyond exact words

**How Transformer Retrieval Works:**
1. **Encode**: Transform text into dense vector representations
2. **Compare**: Use cosine similarity between query and document vectors
3. **Retrieve**: Find documents with highest semantic similarity

**From Sequence Models to Transformers:**
- **RNNs/LSTMs**: Sequential processing, limited context
- **Transformers**: Parallel processing, global attention, rich context

I've also provided these enhancements scripts to the Author of the Headroom as an open-source contributions..[LINK]

**Natural Questions** dataset is used which consists of real Google search queries with Wikipedia answers.
This dataset provides a realistic evaluation of how different retrieval methods handle real-world information needs. More details about the data set can be found here https://pypi.org/project/beir/0.1.1/

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



Maintainer seems to be actively working on Langchain fixes as seen in the commit https://github.com/headroomlabs-ai/headroom/commit/5d025f7a03870a402918e6425ca9fcd40400edb2

Conclusion: Given the issues highlighted in my analyzes I would be hasistant to use headroom package for production systems or as a third party library. This package seems to be still on work in progress and not well configured by looking at the commits as well. 


In unified retrieval comparison notebook, BM25 algorithm does not retrieve correct document for the given query;
    query = "Which Doctor performed Caesarean?" 
Headroom's BM25 and Transformers model retrieves correct document. 

Below is the adapted version of Example 2 from Headroom Langchain docs using the extensions developed from this report. This is also part of the enhancement suggested to the maintainer of headroom. 

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

There is a tradeoff of computational time in using transformer compressor with accuracy performance boost. Transformer retrieval approach provides huge flexibility. Currently Mini-LLM is being used. We can switch models. In this kaggle notebook, you can find the performance of the following models.
MiniLM-L6 (Small & Fast): Lightweight model, good efficiency
BERT-base (Medium): Classic transformer, balanced performance
BGE-small (Retrieval-Optimized): Specifically trained for retrieval tasks

Model Architecture Comparison
MiniLM-L6-v2 (Small & Fast)
Size: 23M parameters, 6 layers
Training: Knowledge distillation from BERT
Strength: Speed and efficiency
Use Case: Real-time applications, resource-constrained environments
BERT-base/DistilBERT (Balanced)
Size: 67M parameters, 12 layers (6 for DistilBERT)
Training: Masked language modeling
Strength: Reliable general-purpose performance
Use Case: Standard NLP tasks requiring proven architecture
BGE-small (Retrieval-Optimized)
Size: 33M parameters
Training: Contrastive learning on query-document pairs
Strength: Optimized for retrieval and similarity
Use Case: Search and semantic retrieval tasks
Key Takeaway
Transformer architectures make different trade-offs between size, speed, and task-specific performance. Smaller models can match or exceed larger ones when optimized for specific tasks.



We can also fine-tune the model. Here is the notebook that fine-tunes the transformers MiniLLM.


Transformers are the neural network architecture built around a self-attention mechanism that weighs the relevance of every token to every other token, regardless of distance that powers most modern language models. Transformers generate the embeddings from text tokens.


Embeddings are numerical vector representations of data (words, sentences, images) where items with similar meaning end up close together in a high-dimensional space, typically learned by a neural network during training. We can use Cosine similarity to numerically score similarity between two vector embeddings. Cosine Similarity measures the angle between vectors ranging from -1 to 1 rather than their magnitude, so a score near 1 means the vectors point in nearly the same direction and represent similar content. Cosine similarity is used to compare those embeddings for tasks like semantic search, recommendations, or retrieval-augmented generation (RAG).


Reference:

[1] Udacity project course