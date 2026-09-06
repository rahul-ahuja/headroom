## Overview 🎯

The purpose is to understand the headroom's main functionalities. A new useful feature has been suggested in `ANALYSIS.md` and implemented in `src` in this subdirectory. `ANALYSIS.md` summarizes the folloiwng;
- The most interesting features you were able to exercise
- Your extension 
- Your evaluation

## Subdirectory Structure
```
headroom/kth_task/
├── src/                                        # Core retrieval modules
│   ├── data_loader.py                          # BeIR/Natural Questions dataset handler
│   ├── bm25_retriever.py                       # Traditional keyword-based retrieval
│   ├── transformer_retriever.py                # Semantic transformer retrieval
│   ├── evaluator.py                            # IR metrics calculation
│   ├── headroom_bm25.py                        # BM25 with HeadroomDocumentCompressor base class
│   ├── headroom_transformer.py                 # Transformer with HeadroomDocumentCompressor base class
│   └── utils.py                                # Utility functions
├── tests/                                      # Unit test suite for validation
│   ├── test_bm25_retriever.py                  # BM25 retriever tests
│   ├── test_transformer_retriever.py           # Transformer retriever tests  
│   └── test_evaluator.py                       # IR metrics evaluator tests
├── notebooks/
│   ├── unified_retrieval_comparison.ipynb      # Main analysis notebook
│   ├── semantic-transformer.ipynb              # Performance comparison of transformer models
│   ├── finetuning-transformer-retrieval.ipynb  # Finetuning of the transformer
│   ├── deepseek_demo.ipynb                     # Running Headroom's main functionality with Deepseek model
│   └── qwen-demo-kaggle.ipynb                  # Running Headroom's main functionality with Qwen model
├── dataset/                                    # Natural Questions test dataset
└── requirements.txt                            # All dependencies
```

## Quick Start Guide

### **🔧 Environment Setup **

`pip install -r requirements.txt`

### **🧪 Validating the implementation**

```bash

# Test individual components
python tests/test_bm25_retriever.py        # BM25 keyword search implementation
python tests/test_evaluator.py             # IR metrics calculation
python tests/test_transformer_retriever.py  # Transformer semantic search

# Final validation 
python -m pytest tests/ -v
```

### **📈 Run demos and analysis **
```
cd kth_task/notebooks
jupyter notebook
```