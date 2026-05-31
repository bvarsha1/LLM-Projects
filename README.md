# LLM Projects

A collection of Large Language Model (LLM) tools and applications, focusing on data generation, efficiency, and developer utility.

## 1. Semantic Pluggable Cache
An intelligent caching mechanism designed for LLM applications. Instead of relying on exact string matches, it leverages vector embeddings to identify and serve semantically similar queries, significantly reducing API costs and latency.

* **Key Features:**
    * Semantic similarity matching using vector embeddings.
    * Pluggable architecture easily integrated into existing LLM pipelines.
    * Configurable distance thresholds for cache hits.
* **Status:** In Development
* **Project Directory:** [Semantic Cache](https://github.com/bvarsha1/LLM-Projects/tree/main/semantic-cache)

---

## 2. Synthetic Docs Generator
An automated tool designed to generate comprehensive, high-quality documentation from source code, unstructured notes, or minimal prompts using LLMs.

* **Key Features:**
    * Context-aware documentation generation.
    * User-friendly interface for seamless interaction.
* **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/thefrugaltechie/synthetic-docs-generator)
* **Project Directory:** [Synthetic Documents Generator](https://github.com/bvarsha1/LLM-Projects/tree/main/synthetic-docs-generator-space)

---

## 3. Synthetic JsonL Data Generator
A specialized data engineering tool built to generate diverse, high-quality synthetic datasets in `.jsonl` format. Perfect for fine-tuning LLMs, training custom models, or creating robust test suites.

* **Key Features:**
    * Generates structured JSON Lines data based on custom schemas.
    * Ideal for preparing fine-tuning datasets for models like GPT, Llama, and Mistral.
* **Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/thefrugaltechie/synthetic-data-generator)
* **Project Directory:** [Synthetic Data Generator](https://github.com/bvarsha1/LLM-Projects/tree/main/synthetic-data-generator-space)

---

## 4. EasyRAG
A streamlined, highly modular Retrieval-Augmented Generation (RAG) framework designed to effortlessly bridge personal or enterprise knowledge bases with LLMs. It abstracts the complexities of data ingestion, retrieval optimization, and pipeline benchmarking into a developer-friendly interface.

* **Key Features:**
    * **Mix & Match Strategies:** Highly flexible architecture allowing developers to experiment with and combine different data ingestion and retrieval strategies.
    * **Hybrid Search:** Combines semantic vector search with traditional keyword search (BM25) out of the box for highly precise document retrieval.
    * **Advanced RAG Capabilities:** Implements sophisticated retrieval techniques to minimize context noise and prevent LLM hallucinations.
    * **Built-in Performance Evaluation:** Includes native benchmarking tools to evaluate the performance, latency, and accuracy of the RAG pipeline.
* **Status:** In Development

---
