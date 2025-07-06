# CrediTrust AI Complaint Analysis Architecture

## Overview
The system is designed to process financial customer complaints, extract insights, and provide actionable recommendations through a RAG (Retrieval-Augmented Generation) pipeline.

## Components

### 1. Data Pipeline
- Ingests raw complaint data from multiple sources
- Cleans and preprocesses text
- Enriches with metadata
- Stores in processed format

### 2. Embedding Service
- Uses hybrid embedding model combining:
  - Text semantic embeddings (Sentence Transformers)
  - Sentiment embeddings (BERTweet)
  - Temporal features
- Generates dense vector representations for retrieval

### 3. Retrieval System
- Hierarchical retrieval process:
  1. Initial vector similarity search
  2. Query-type specific filtering
  3. Optional graph-based expansion
- Supports multiple query types (trends, specific issues, comparisons)

### 4. Generation Service
- Uses OpenAI GPT models for analysis
- Provides:
  - Summary insights
  - Root cause analysis
  - Recommended actions
- Supports streaming for real-time updates

### 5. Analytics Dashboard
- Visualizations for:
  - Complaint trends over time
  - Sentiment distribution
  - Product comparison
  - Issue clustering