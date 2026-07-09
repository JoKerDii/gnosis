---
title: 2025 March - What I Have Read
date: 2025-03-04 00:15:05
tags: readings
---

## Blogs and Articles

> **How to Build a Graph RAG App - Steve Hedden** [[Link](https://towardsdatascience.com/how-to-build-a-graph-rag-app-b323fc33ba06/)]

Walking through building a graph rag app that improves LLM accuracy using knowledge graphs. It covers data preparation, search refinement with MeSH terms, and article summarization. 

> *We believe that, in 2025, we may see the first AI agents “join the workforce” and materially change the output of companies. We continue to believe that iteratively putting great tools in the hands of people leads to great, broadly-distributed outcomes.*
>
> **― Reflections - Sam Altman** [[Link](https://blog.samaltman.com/reflections)]

> **Structured Report Generation Blueprint with NVIDIA AI** [[Link](https://blog.langchain.dev/structured-report-generation-blueprint/)] [[YouTube](https://www.youtube.com/watch?v=E04rFNtwFcA)]

> **Sky-T1: Train your own O1 preview model within $450** - [NovaSky](https://novasky-ai.github.io/) [[Link](https://novasky-ai.github.io/posts/sky-t1/)]

> **Agents - Chip Huyen** [[Link](https://huyenchip.com//2025/01/07/agents.html)]

This guide provides a comprehensive exploration of AI-powered agents, focusing on their capabilities, planning, tool selection, and failure modes. It delves into the factors determining an agent's performance, how LLMs can plan, and how to augment planning capabilities. It also provides insights into agent failures and how to evaluate them effectively.

> **The Batch Issue 284 - DeepLearning.AI - Andrew Ng** [[Link](https://www.deeplearning.ai/the-batch/issue-284/)]

Andrew Ng highlights AI Product Management’s growth as software becomes cheaper to build.

> **DeepSeek V3 LLM NVIDIA H200 GPU Inference Benchmarking - DataCrunch** [[Link](https://datacrunch.io/blog/deepseek-v3-llm-nvidia-h200-gpu-inference-benchmarking)]

> **Global-batch load balance almost free lunch to improve your MoE LLM training - Qwen** [[Link](https://qwenlm.github.io/blog/global-load-balance/)]

MoE models struggle with expert underutilization due to micro-batch-level load balancing, which fails when data within a batch lacks diversity. This results in poor expert specialization and model performance.

The paper proposes global-batch load balancing, where expert selection frequencies are synchronized across all parallel groups, ensuring more effective domain specialization and improved performance. 

Global-batch load balancing outperforms micro-batch balancing in all tested configurations. It shows improved performance and expert specialization, with models achieving better results across various data sizes and domains.

> **How to Evaluate LLM Summarization - Isaac Tham** [[Link](https://towardsdatascience.com/how-to-evaluate-llm-summarization-18a040c3905d/)]

A quantitative, research-backed framework for evaluating LLM summaries, focusing on conciseness and coherence. This guide explores challenges in summarization evaluation, defines key quality metrics (conciseness, coherence), and improves the Summarization Metric in the DeepEval framework. Includes a GitHub notebook for applying these methods to assess summaries of long-form content systematically.

> **We just gave sight to smolagents - HuggingFace** [[Link](https://huggingface.co/blog/smolagents-can-see)]

This tutorial is about how to integrate vision capabilities into autonomous agents using smolagents. It explains passing images to agents in two ways: at initialization or dynamically via callbacks. It demonstrates building a web-browsing agent with vision using the MultiStepAgent class and helium. The agent performs actions like navigation, popup handling, and dynamic webpage analysis.

> **On DeepSeek and Export Controls - Dario Amodei** [[Link](https://darioamodei.com/on-deepseek-and-export-controls)]

Highlighting export controls' impact on AI geopolitics.

> **Workflows and Agents - LangGraph** [[Link](https://langchain-ai.github.io/langgraph/tutorials/workflows/)]

Review of common patterns for agentic systems.

> **Constitutional Classifiers: Defending against universal jailbreaks - Anthropic** [[Link](https://www.anthropic.com/research/constitutional-classifiers)] [[YouTube](https://youtu.be/BaNXYqcfDyo)]

Anthropic invites everyone to test its new safety classifier that eradicates jailbreaks and further increases Claude's over-refusal rate.

> **Open-source DeepResearch – Freeing our search agents - HuggingFace** [[Link](https://huggingface.co/blog/open-deep-research)]

Hugging Face challenges OpenAI’s Deep Research with an open-source alternative, beating previous SOTA by 9 points.

> **Choosing the Right AI Agent Framework: LangGraph vs CrewAI vs OpenAI Swarm - Yi Zhang** [[Link](https://blog.relari.ai/choosing-the-right-ai-agent-framework-langgraph-vs-crewai-vs-openai-swarm-56f7931b4249)]

Compare LangGraph, CrewAI, and OpenAI Swarm frameworks for building agentic applications with hands-on examples. Understand when to use each framework, and get a preview of debugging and observability topics in Part II.

> **How to Scale Your Model - Google DeepMind** [[Link](https://jax-ml.github.io/scaling-book/)]

Learn how to scale LLMs on TPUs by understanding hardware limitations, parallelism, and efficient training techniques. Explore how to estimate training costs, memory needs, and optimize performance using strategies like data, tensor, pipeline, and expert parallelism. Gain hands-on experience with LLaMA-3, and learn to profile and debug your code.

> **Three Observations - Sam Altman** [[Link](https://blog.samaltman.com/three-observations)]

Sam outlines AI trends: AI’s scaling limits, cost reduction, and the future of autonomous agents.

> **How to deploy and fine-tune DeepSeek models on AWS - HuggingFace** [[Link](https://huggingface.co/blog/deepseek-r1-aws)]

Deploy and fine-tune DeepSeek-R1 models on AWS using Hugging Face with GPUs, SageMaker, and EC2 Neuron.

> **Building a Universal Assistant to connect with any API - Pranav Dhoolia** [[Link](https://medium.com/heurislabs/building-a-universal-assistant-to-connect-with-any-api-89d7c353e524)]

Convert any OpenAPI spec into an MCP-compatible API assistant without writing custom integration code. Use a generic MCP server to expose API endpoints dynamically. This approach simplifies integration, expands compatibility, and makes scaling API support more efficient.

> **From PDFs to Insights: Structured Outputs from PDFs with Gemini 2.0 - Philschmid** [[Link](https://www.philschmid.de/gemini-pdf-to-data)] 

Learn to convert PDFs into structured JSON using Gemini 2.0. Set up the SDK, process files, manage tokens, and define JSON schemas with Pydantic. Covers real-world examples like invoices and forms, best practices, and cost management, works within the free tier.

> **The Hidden Ways We Really Work Together - Microsoft** [[Link](https://www.microsoft.com/en-us/worklab/patterns-hidden-inside-the-org-chart)]

> **Managing LLM implementation projects - Piotr Jurowiec** [[Link](https://piotr-jurowiec.medium.com/managing-llm-implementation-projects-2daa0ef5c67a)]

Discover how to implement LLMs from initial planning to deployment. Establish project goals, select suitable architectures, preprocess data, train and evaluate models, optimize hyperparameters, and incorporate domain expertise. Tackle challenges such as hallucinations, security risks, regulatory compliance, and scalability limitations. Develop systematic workflows for building and managing LLM-based applications.

> **How to build a ChatGPT-Powered AI tool to learn technical things fast - AWS** [[Link](https://community.aws/content/2hEttGRiSiGwAn2Bzl0lbQcT7v6/build-ai-with-hf-chatgpt)]

> **What Problem Does The Model Context Protocol Solve? - AIhero** [[Link](https://www.aihero.dev/what-problem-does-model-context-protocol-solve)]

Learn how the Model Context Protocol (MCP) simplifies integrating large language models (LLMs) with external APIs.

MCP acts as a connector between LLMs and external data sources, facilitating interactions with tools without requiring LLMs to understand intricate APIs. By providing a standardized interface, it streamlines integrations with platforms like GitHub, enhancing workflow speed and efficiency.

> **Most AI value will come from broad automation, not from R&D - Epoch AI** [[Link](https://epoch.ai/gradient-updates/most-ai-value-will-come-from-broad-automation-not-from-r-d)]

Epoch AI's article argues against the popular notion that the primary economic benefit of artificial intelligence will stem from its application in research and development. Instead, the authors posit that AI's most significant value will arise from its widespread deployment in automating existing labor across various sectors. 

## Substack

> **Tencent: Betting Big on AI - App Economy Insights** [[Link](https://www.appeconomyinsights.com/p/tencent-betting-big-on-ai)]

![tencent_corporate_overview](2025-March/tencent_corporate_overview.png)

Tencent's proprietary HunYuan framework has developed into a central AI platform catering to both consumers and enterprises. Originally centered on text and conversational AI, HunYuan has expanded to support multimodal capabilities, including image, video, and 3D generation, where it has attained top rankings in industry benchmarks.

![HunYuan_Thesis](2025-March/HunYuan_Thesis.png)

Tencent has a Dual-Core AI strategy: It combines its proprietary T1 model with external AI, such as DeepSeek’s R1, in a “double-core” approach. Yuanbao chatbot utilizes both—T1 for deep reasoning and R1 for quick responses—while WeChat Search enhances accuracy by integrating T1 with DeepSeek.

![tencent_multi_model_strategy](2025-March/tencent_multi_model_strategy.png)

> **Google: Biggest Deal Ever - App Economy Insights** [[Link](https://www.appeconomyinsights.com/p/google-biggest-deal-ever)]

Alphabet has completed its largest acquisition to date with a $32 billion deal to acquire cloud security startup Wiz. If successful, this move could redefine GCP’s security portfolio, strengthening its stance as AI-driven cloud computing becomes the focal point.

![google_biggest_acquisition](2025-March/google_biggest_acquisition.png)

## Papers and Reports

> **Whitepaper Agents - Authors: Julia Wiesinger, Patrick Marlow and Vladimir Vuskovic** [[Link](https://www.kaggle.com/whitepaper-agents)]

Google’s whitepaper explains how AI agents use reasoning, tools, and external data to automate tasks, turning large language models (LLMs) into workflow automation systems. Google suggests using **LangChain** for prototyping and **Vertex AI** for scaling production-ready agents. its framework provides a standardized approach to ensure reliable AI agent execution.

**Key Components**

1. **Decision Engine** – The LLM plans and executes tasks using reasoning methods like ReAct or Chain-of-Thought.
2. **Tool Integration** – Agents interact with APIs, databases, and real-time data.
3. **Orchestration Layer** – Manages task execution and decision-making.

**Tool Types**

1. **Extensions** – Directly call APIs for automation.
2. **Functions** – Allow developers to control execution.
3. **Data Stores** – Use retrieval-augmented generation (RAG) for external data access.

**Use Cases**

Agents handle tasks like personalized recommendations, workflow automation, and database queries. For example, they can fetch a user’s purchase history and generate tailored responses. 

> **Introducing *smolagents*, a simple library to build agents - HuggingFace** [[Link](https://huggingface.co/blog/smolagents)]

> **Memory Layers at Scale - Meta** [[Link](https://arxiv.org/abs/2412.09764)]

> **2 OLMo 2 Furious - Alien AI** [[Link](https://arxiv.org/abs/2501.00656)]

> **HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs** [[Link](https://arxiv.org/abs/2412.18925)]

This paper shows how to build domain-specific reasoning models using a two-stage training process. HuatuoGPT-o1, a medical LLM, enhances complex reasoning using this two-stage approach: (1) supervised fine-tuning (SFT) with complex Chain-of-Thought (CoT) and (2) reinforcement learning (RL) using a verifier to refine reasoning. 

![huatuogpto1](2025-March/huatuogpto1.png)

> **Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps - Google DeepMind** [[Link](https://arxiv.org/abs/2501.09732)]

Google DeepMind introduces noise search method, outperforming traditional denoising in diffusion models.

> **Chain of Agents: Large language models collaborating on long-context tasks - Google Research** [[Link](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/)] [[Paper](https://openreview.net/pdf?id=LuCLf4BJsr)]

> **SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training - Google DeepMind** [[Link](https://tianzhechu.com/SFTvsRL/)] [[Link](https://arxiv.org/abs/2501.17161v1)]

Explaining why reinforcement learning outperforms supervised fine-tuning for model generalization.

> **Learning to Plan & Reason for Evaluation with Thinking-LLM-as-a-Judge** [[Link](https://arxiv.org/abs/2501.18099)]

A new preference optimization algorithm for LLM-as-a-Judge models.

> **Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks** [[Link](https://arxiv.org/abs/2501.13946v1)]

Generative AI models often produce hallucinations, making them less reliable and reducing trust in AI systems. In this work, a multi-agent system is designed using over 300 prompts to induce hallucinations. AI agents at different levels review and refine outputs using distinct language models, structured JSON communication, and the OVON framework for seamless interaction. New KPIs are introduced to measure hallucination levels.

> **ELEGNT: Expressive and Functional Movement Design for Non-Anthropomorphic Robot - Apple** [[Link](https://machinelearning.apple.com/research/elegnt-expressive-functional-movement)] [[Link](https://arxiv.org/abs/2501.12493)]

This is very cool.

> **π0 and π0-FAST: Vision-Language-Action Models for General Robot Control - Hugging Face** [[Link](https://huggingface.co/blog/pi0)]

Hugging Face publishes the first open-source robotics foundation models for real-world applications.

> **Claude’s extended thinking - Anthropic** [[Link](https://www.anthropic.com/research/visible-extended-thinking)]

Claude 3.7 Sonnet introduces extended thinking, visible reasoning, and improved agentic capabilities for complex tasks.

> **Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers** [[Link](https://arxiv.org/abs/2409.04109)]

This study evaluated the ability of LLMs to generate novel, expert-level research ideas compared to human experts by recruiting over 100 NLP researchers for idea generation and blind reviews. Results showed that LLM-generated ideas were rated as more novel than human ideas (p < 0.05) but slightly less feasible. While LLMs demonstrated promising ideation capabilities, challenges such as limited idea diversity and unreliable self-evaluation were identified, highlighting areas for improvement in developing effective research agents.

> **Transformers without Normalization** [[Link](https://jiachenzhu.github.io/DyT/?utm_source=alphasignal)]

Yann LeCun and his team have proposed Dynamic Tanh (DyT) as an alternative to conventional normalization layers in deep learning models. This innovative method, leveraging the scaled tanh function, delivers performance on par with or superior to techniques like LayerNorm and RMSNorm. Notably, its ability to lower computational costs while preserving model efficiency makes it particularly compelling.

> **Energy** [[Link](https://infrastructurereportcard.org/wp-content/uploads/2025/03/Energy.pdf)]

Most serious issues:

* Aging and overburdened energy infrastructure is the most serious issue. 
* Energy demand is at its highest growth rate in 20 years. EV adoption and AI workloads are accelerating the strain on the grid.
* There is increased frequency of extreme weather events causing outages.
* The U.S. experienced twice as many weather-related power outages from 2014–2023 compared to 2000–2009.

Some key trends: 

* Energy demand is rising rapidly, especially due to data centers, AI, and EV adoption.
* Extreme weather is causing more frequent and severe power outages.
* The transition to renewables is accelerating, but grid interconnection delays are slowing progress.
* Grid infrastructure is aging and requires massive investment, but funding gaps remain.
* Transformer shortages and long lead times are hindering grid expansion and maintenance.
* Cybersecurity threats and physical attacks on substations are emerging risks.

## YouTube and Podcasts

> **Fixing the American Dream with Andrew Schulz - All-In Podcast** [[Link](https://www.youtube.com/watch?v=IjU-Nd6iiQ4&ab_channel=All-InPodcast)]

> **E179｜DeepSeek技术解析：为何引发英伟达股价下跌？- 硅谷101播客** [[Link](https://www.youtube.com/watch?v=f9Wm1ayTKlQ&ab_channel=%E7%A1%85%E8%B0%B7101%E6%92%AD%E5%AE%A2)]

> **This Year in Uber’s AI-Driven Developer Productivity Revolution - Gradle** [[Link](https://www.youtube.com/watch?v=jp-fBw07r7c&t=2s)]

> **GraphRAG: The Marriage of Knowledge Graphs and RAG: Emil Eifrem - AI Engineer** [[Link](https://www.youtube.com/watch?v=knDDGYHnnSI)]

Great intro to Graph RAG from Prof Emil Eifrem. Check out [neo4j genai ecosystem](https://neo4j.com/labs/genai-ecosystem/).

Some articles mentioned: "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering" [[Link](https://arxiv.org/abs/2404.17723v1)], and "GraphRAG: Unlocking LLM discovery on narrative private data" [[Link](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)].

> **Building a fully local "deep researcher" with DeepSeek-R1 - LangChain** [[Link](https://www.youtube.com/watch?v=sGUjmyfof4Q)]

This tutorial reviews DeepSeek R1's training methods, explains downloading the model via Ollama, and demonstrates JSON-mode testing. Test its local "deep research" assistant, which performs web research and iterative summarization with reflection for improved results.

> **Building Effective Agents with LangGraph - LangChain** [[Link](https://www.youtube.com/watch?v=aHCDrAbH_go)]

This video shows the difference between agents and workflows and when to use each. You'll Implement patterns like prompt chaining, parallelization, and routing using LangGraph. The session covers building agents, applying advanced patterns, and understanding how LangGraph enhances automation and optimization in AI systems.

> **Nvidia's GTC 2025 Keynote: Everything Announced in 30 Minutes - Amrit Talks** [[Link](https://www.youtube.com/watch?v=DT5IMxzfdL0&ab_channel=AmritTalks)]

> **How I use LLMs - Andrej Karpathy** [[Link](https://www.youtube.com/watch?v=EWvNQjAaOHw&ab_channel=AndrejKarpathy)]

> **White House BTS, Google buys Wiz, Treasury vs Fed, Space Rescue - All-In Podcast** [[Link](https://www.youtube.com/watch?v=p1MAA8y4CgU&ab_channel=All-InPodcast)] 

> **Satya Nadella – Microsoft’s AGI Plan & Quantum Breakthrough - Dwarkesh Partel** [[Link](https://www.youtube.com/watch?v=4GLSzuYXh6w&ab_channel=DwarkeshPatel)]

Satya Nadella discusses AI, AGI skepticism, economic growth, quantum computing, AI pricing, gaming models, and legal challenges. Notes of insights impressed me:

1. Nadella believes hyperscalers (like Microsoft Azure, AWS, and Google Cloud) will be major beneficiaries of AI advancements. The exponential growth in compute demand for AI workloads—both for training and inference—will drive massive infrastructure needs. Hyperscalers are well-positioned to meet this demand due to their ability to scale compute, storage, and AI accelerators efficiently.
2. He argues that hyperscale infrastructure is not a winner-takes-all market. Enterprises and corporations prefer multiple suppliers to avoid dependency on a single vendor. This structural dynamic ensures competition and prevents monopolization.
3. While there may be a few dominant closed-source AI models, Nadella predicts that open-source alternatives will act as a check, preventing any single entity from monopolizing the AI model space. He draws parallels to the coexistence of closed-source (e.g., Windows) and open-source systems in the past.
4. He highlights that governments worldwide are unlikely to allow private companies to dominate AI entirely. Regulatory and state involvement will likely shape the landscape, further preventing a winner-takes-all scenario.
5. In consumer markets, network effects can lead to winner-takes-all dynamics (e.g., ChatGPT's early success). However, in enterprise markets, multiple players will thrive across different categories.
6. He disagrees with the notion that AI models or cloud infrastructure will become commoditized. At scale, the complexity of managing hyperscale infrastructure and the know-how required to optimize it create significant barriers to entry and sustain profitability.
7. Microsoft aims to build a versatile hyperscale fleet capable of handling large training jobs, inference workloads, and specialized tasks like reinforcement learning (RL). The company focuses on distributed computing, global data center placement, and high utilization of resources to meet diverse AI demands.
8. Nadella envisions a future where AI agents and specialized models will drive even greater compute demand. He emphasizes the importance of building infrastructure that can support both training and inference at scale, while also accommodating evolving AI research and development.
9. Microsoft Research (MSR) has a history of investing in fundamental, curiosity-driven research, often with no immediate payoff. Nadella emphasizes the importance of maintaining this culture, even if the benefits may only materialize decades later. Nadella highlights the difficulty of transitioning from research breakthroughs to scalable products. The role of leadership is to ensure that innovations are not only technically sound but also commercially viable.
10. Nadella envisions quantum computing being accessed via APIs, similar to how cloud services are used today. This could democratize access to quantum capabilities for research and industry.

> **Instrumenting & Evaluating LLMs - Hamei Husain** [[Link](https://www.youtube.com/watch?v=SnbGD677_u0&ab_channel=HamelHusain)]

> **How to Get Your Data Ready for AI Agents (Docs, PDFs, Websites) - Dave Ebbelaar** [[Link](https://www.youtube.com/watch?v=9lBTS5dM27c&ab_channel=DaveEbbelaar)]

> **The AI Cold War, Signalgate, CoreWeave IPO, Tariff Endgames, El Salvador Deportations - All-In Podcast** [[Link](https://www.youtube.com/watch?v=Iazo7g40VbQ&ab_channel=All-InPodcast)]

## Github

> **a smol course - HuggingFace** [[Link](https://github.com/huggingface/smol-course)]

Good course to help you to align LLM with specific use cases. It includes instruction tuning, preference alignment using DPO/ORPO, LoRA, prompt tuning, and multimodal model adaptation, and it covers creating synthetic datasets, evaluation, and efficient inference. 

> **AI Hedge Fund** [[Link](https://github.com/virattt/ai-hedge-fund)]

POC for educational purposes.

> **build-your-own-x** [[Link](https://github.com/codecrafters-io/build-your-own-x)]

Interesting resources / tutorials for building technologies from scratch to deepen practical understanding.

> **Open R1- HuggingFace** [[Link](https://github.com/huggingface/open-r1)]

A fully open reproduction of DeepSeek-R1.

> **AI Tutor Using RAG, Vector DB, and Groq** [[Link](https://colab.research.google.com/drive/1ZdVuxLTEkQJrV6EdZwEqtvoRkGF4FOY0)]

> **GenAI Agents: Comprehensive Repository for Development and Implementation** [[Link](https://github.com/NirDiamant/GenAI_Agents)]

> **DeepSeek Open Infra** [[Link](https://github.com/deepseek-ai/open-infra-index/tree/main)]

> **Awesome MCP Servers** [[Link](https://github.com/punkpeye/awesome-mcp-servers)]

This repository provides access to a selection of curated Model Context Protocol (MCP) servers designed for seamless AI model-resource interaction. It features both production-ready and experimental servers, offering capabilities like file access, database connections, and API integrations. There are frameworks, tutorials, and practical tips to enhance model deployment and maximize resource efficiency in real-world applications.

## News 

> **Replit Integrates xAI** [[Link](https://docs.replit.com/updates/2025/03/07/changelog#xai-integration)]

> **Expand your research with proprietary financial data - Perplexity** [[Link](https://www.perplexity.ai/enterprise/crunchbase-factset-integration)]

Crunchbase, FactSet, and DeepSeek R1 now power enterprise insights.

> **Google to acquire cloud security startup Wiz for $32 billion after deal fell apart last year - CNBC** [[Link](https://www.cnbc.com/2025/03/18/google-to-acquire-cloud-security-startup-wiz-for-32-billion.html)]

> **Leave it to Manus - Manus** [[Link](https://manus.im/)]
