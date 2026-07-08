---
title: Getting Started with the Knowledge Agent
tags: [example, getting-started]
---

# Getting Started

This is a sample Markdown note. Drop your own `.md` files into this folder
(or point `KNOWLEDGE_DIR` at another directory) and run:

    python -m src.cli index
    python -m src.cli search "how do I get started?"

The agent parses front-matter metadata, splits the body into overlapping
chunks, embeds each chunk with a local Sentence-Transformers model, and stores
the vectors in LanceDB for fast semantic search — all fully offline.
