# Medical RAG Classifier

Retrieval-Augmented Generation system for automated quality control of AI services in radiology and fluorography.

## Overview

This system combines semantic search (Sentence Transformers), lexical search (BM25), and generative language models (DialoGPT) to classify radiology reports as "pathology" or "normal" based on established medical criteria.

## Features

- **Hybrid Search**: Combines semantic and lexical search for optimal retrieval
- **Medical Criteria**: Based on established radiological standards
- **Explainable AI**: Provides detailed explanations for classifications
- **Batch Processing**: Efficient processing of large datasets
- **Multilingual Support**: Optimized for Russian medical texts

## Installation

```bash
git clone https://github.com/your-username/medical-rag-classifier.git
cd medical-rag-classifier
pip install -r requirements.txt
