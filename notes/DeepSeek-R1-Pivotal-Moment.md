---
title: 'DeepSeek-R1: Pivotal Moment'
date: 2025-02-01 21:21:58
tags: knowledge
---

My thoughts regarding the AI landscape at the current stage:

As open-source AI becomes more affordable, it is poised to become as ubiquitous and accessible as electricity—financially viable for everyone. The AI and AGI arms race, whether between nations, open- and closed-source models, or competing companies, is effectively over or should be over, and the outcome is clear. Compute power still remains essential, but semiconductor giants like NVIDIA should look beyond language model training and inference, shifting their focus to the next frontiers, such as robotics and world models. Now is the time for developers and startups to concentrate on the vertical integration of AI, where real economic value can be realized.

## DeepSeek - Background

DeepSeek began as a research offshoot of **High-Flyer**—a hedge fund that had already amassed a large GPU inventory (reportedly 10,000 Nvidia A100s in 2021). Over time, this resource base appears to have grown, with estimates suggesting that—when you account for research, ablation experiments, and shared infrastructure with trading—the effective pool might be closer to 50,000 GPUs. This expansive compute power enables DeepSeek to run many experiments simultaneously and quickly iterate on new architectures.

By leveraging a shared infrastructure with its hedge fund operations, DeepSeek can reinvest profits from quant trading into AI research. This model of “doing more with less” not only challenges the notion that massive, multibillion-dollar compute expenditures are necessary to build world-class AI models but also has broader implications for the industry. It raises questions about the future economics of AI development and the potential for more cost-efficient, research-driven models to shift market dynamics, as seen by the notable impact on Nvidia’s stock and market sentiment.

## Export Controls on GPUs to China

In essence, the U.S. government originally imposed limits on chips that exceed certain thresholds in both **interconnect bandwidth** and **compute (FLOPs)** to restrict China’s ability to train massive AI models. Early on, chips that combined high interconnect speeds with high FLOPs were off‐limits.

For example, the H100—one of Nvidia’s top GPUs—was deemed too powerful. In response, Nvidia developed the H800, which maintained the same floating point performance (FLOPs) as the H100 but had its interconnect bandwidth intentionally reduced to meet U.S. export criteria. However, when the government later decided to tighten controls further (targeting chips solely on FLOPs), even the H800 was banned. This led Nvidia to innovate once again with the H20, a chip that now offers full interconnect bandwidth (and even improved memory characteristics over the H100) but with a deliberate cut in overall FLOPs to satisfy export rules.

The strategic rationale behind these controls is to “decap” China’s compute—especially for large-scale AI training—by limiting how many of the most advanced GPUs (and thus the overall density of compute) can be legally acquired. While Chinese companies can still purchase GPUs to train models, the overall capacity available for training (which is critical for developing super-powerful AI) is being capped. This is seen as a way to maintain U.S. and allied leadership in AI, particularly in a world where super-powerful AI may soon offer decisive military and economic advantages.

### Sidenote - GPUs for AI

**Keys GPU Specifications:**

* **FLOPS (Compute Power)**: Critical for training large models (e.g., GPT-4) but less critical for inference tasks like reasoning.
* **Memory Bandwidth/Capacity**: Determines how much data (e.g., KV cache in transformers) can be stored and accessed quickly, crucial for long-sequence tasks.
* **Interconnect Speed**: Affects communication between GPUs in clusters, important for distributed training but less regulated now.

**H20 vs. H100: Tradeoffs for AI Workloads:**

- **H20 (China-Specific):** has its strength in higher memory bandwidth and capacity than H100, making it better suited for **reasoning tasks** (e.g., long-context inference, chain-of-thought). However, FLOPS (≈1/3 of H100 on paper, ≈50-60% in practice) is reduced, limiting its utility for training.

  **Regulatory Context**: Designed to comply with U.S. export controls that focus on FLOPS, allowing Nvidia to ship 1M units to China in 2023 (20-25% of total GPUs).

- **H100**: Optimized for FLOPS-heavy training but less efficient for memory-bound inference tasks

**Why Memory Matters for Reasoning:**

- **KV Cache in Transformers** stores keys/values of all tokens in a sequence for attention mechanisms. Memory demands grow **quadratically** with sequence length (e.g., 10K+ tokens in reasoning tasks).
- **Autoregressive Generation**: Output tokens require sequential processing, forcing repeated KV cache access. This limits parallelism and increases memory pressure. Tasks like agentic AI or chain-of-thought involve generating long outputs (10K+ tokens), stressing memory bandwidth/capacity. 

## DeepSeek - Technical Comments

Paper: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)

**GPU Infrastructure with Nvidia Hardware**

- DeepSeek trains on **Nvidia GPUs**. These are equipped with many cores (organized into streaming multiprocessors, or SMs) that perform the heavy lifting during both training and inference.
- The GPUs they used were those **legally available in China**, which imposed certain limitations—especially on interconnect bandwidth between units. This meant that DeepSeek needed to overcome hardware constraints that might not be present with the very latest high-end GPUs elsewhere.

**Custom Low-Level Optimization**

- Instead of relying solely on Nvidia’s standard NCCL (Nvidia Communications Collectives Library) for handling inter-GPU communications, DeepSeek’s engineers developed custom scheduling techniques. They even **scheduled communications at the SM level**, which is more granular than the typical approach.
- Their implementation involved programming approaches that went deep into the hardware—down to using **PTX** (an intermediate assembly-like language for CUDA). This allowed them to squeeze extra efficiency from each GPU by reducing the overhead in communication between layers of the model.

**Efficiency via Architectural Choices**

- One of the key innovations was using a **sparse Mixture of Experts (MoE) architecture**. With a model that can have hundreds of billions of parameters overall but only activates a fraction (e.g., around 37 billion at a time), the compute and memory demands are dramatically reduced. This architectural choice means that even if the hardware isn’t the absolute latest, it can still be very cost-effective by not needing to run every parameter for every token.
- DeepSeek's novel attention mechanism **MLA (Multi-Head Latent Attention)** reduces memory usage by 80–90% compared to traditional transformer attention. This optimization lowers computational costs, especially for long-context processing, without sacrificing performance.
- By optimizing both the hardware usage (through custom scheduling and low-level programming) and the model architecture (via MoE and MLA), DeepSeek manages to cut down on the cost of training. This is crucial given the significant compute expense associated with large-scale language models.

**Pre-Training and Context Window Extension**

* Pre-trained on 14.8 trillion tokens drawn from a multilingual corpus (primarily English and Chinese) with a higher proportion of math and programming content compared to previous iterations.
* Utilizes a two-phase extension (via the YaRN framework) to expand the context length from 4K tokens to 32K and finally to 128K tokens.
* Reported training cost for V3 is approximately $5.58 million, consuming about 2.788 million GPU-hours on Nvidia H800 GPUs. This figure is significantly lower than the hundreds of millions typically reported by US rivals.

**Post-Training: Supervised Fine-Tuning & Reinforcement Learning**

* V3 is fine-tuned on a carefully curated dataset of approximately 1.5 million examples (both reasoning and non-reasoning tasks) to improve instruction-following and output formatting.
* DeepSeek employs **GRPO**—a **group relative policy optimization method**—to reward outputs based on correctness (accuracy rewards) and presentation (format rewards).
* R1 leverages RL to fine-tune the reasoning process, rewarding chain-of-thought quality and encouraging the model to generate self-reflective “aha moments.”

**Speed-to-Market and Safety Tradeoffs**

* DeepSeek prioritizes rapid deployment over extensive safety testing, avoiding delays and costs associated with ethical reviews (common in Western firms like Anthropic). This "ship-first" approach reduces development cycle expenses.

- Releasing model weights publicly attracts third-party hosting and innovation, indirectly expanding reach without bearing full infrastructure costs.

## **The Tech and Business Perspective**

The release of DeepSeek-R1 marks a pivotal moment in the AI industry, igniting discussions about open-source dominance, market disruption, and geopolitical implications.

**Industry Leaders Weigh In**:

**Yann LeCun (Meta’s Chief AI Scientist)**

LeCun emphasized the growing power of open-source models over proprietary approaches:

> *"To people who see the performance of DeepSeek and think China is surpassing the US in AI. You are reading this wrong. The correct reading is: Open source models are surpassing proprietary ones."*

**Andrej Karpathy (OpenAI Co-founder)**

Karpathy pointed out the continued need for large-scale computing while praising DeepSeek’s efficiency:

> *"Does this mean you don't need large GPU clusters for frontier LLMs? No, but you have to ensure that you're not wasteful with what you have, and this looks like a nice demonstration that there's still a lot to get through with both data and algorithms."*

**Satya Nadella (Microsoft CEO)**

Nadella underscored the significance of DeepSeek, highlighting its role in making AI reasoning more accessible:

> *"We should take the developments out of China very, very seriously."*
>  *"DeepSeek has had some real innovations. … Obviously, now all that gets commoditized."*
>  *"When token prices fall, inference computing prices fall, that means people can consume more, and there will be more apps written."*

**Mark Zuckerberg (Meta CEO)**

Zuckerberg acknowledged DeepSeek's novel infrastructure optimizations:

> *"DeepSeek had a few pretty novel infrastructure optimization advances, which, fortunately, they published them, so we can not only observe what they did, but we can read about it and implement it, so that'll benefit us."*
>  *"Always interesting when there's someone who does something better than you. Let's make sure we are on it."*

**Aravind Srinivas (Perplexity AI CEO)**

Srinivas stressed the importance of foundational innovation:

> *"We need to build, not just wrap existing AI."*

**Marc Andreessen (Andreessen Horowitz Co-founder)**

He likened DeepSeek-R1 to a historic milestone:

> *"DeepSeek R1 is AI's Sputnik moment."*

**Tim Cook (Apple CEO)**

Cook gave a measured response during an earnings call:

> *"In general, I think innovation that drives efficiency is a good thing."*

## **Academic and Research Perspectives**

**AI Researchers on DeepSeek-R1**:

**Timnit Gebru (AI Ethics Researcher)**

Gebru reflected on past AI development priorities:

> *"At Google, I asked why they were fixated on building THE LARGEST model. Why are you going for size? What function are you trying to achieve? They responded by firing me."*

**Ethan Mollick (Wharton AI Professor)**

Mollick focused on accessibility rather than capabilities:

> *"DeepSeek is a really good model, but it is not generally a better model than o1 or Claude. But since it is both free and getting a ton of attention, I think a lot of people who were using free 'mini' models are being exposed to what an early 2025 reasoner AI can do and are surprised."*

**Andrew Ng (AI Researcher and Entrepreneur)**

Ng saw the market reaction as an opportunity for developers:

> *"Today's 'DeepSeek selloff' in the stock market—attributed to DeepSeek V3/R1 disrupting the tech ecosystem—is another sign that the application layer is a great place to be. The foundation model layer being hyper-competitive is great for people building applications."*

**Global Academic Community Response**:

Huan Sun from Ohio State University noted that DeepSeek's affordability is expanding LLM adoption in research. Cong Lu from the University of British Columbia highlighted R1’s rapid adoption, surpassing 3 million downloads on Hugging Face in a week. Meanwhile, safety concerns emerged as studies revealed R1 is 11 times more likely to generate harmful content compared to OpenAI models, prompting calls for better safeguards.

## Impact Discussion

**Market and Industry Impact**

The release of DeepSeek-R1 caused massive shifts in financial markets. U.S. tech stocks collectively lost $\$1$ trillion, with Nvidia suffering record losses due to the rising competition from this cost-efficient model. Investors are recalibrating AI development strategies as DeepSeek achieved comparable performance to OpenAI’s models at just $\$6$ million versus OpenAI’s $\$100$ million.

**Integration into Cloud Ecosystems**

AWS and Microsoft Azure have incorporated DeepSeek-R1, enabling developers to explore its capabilities securely and cost-effectively. The emergence of cost-effective models like DeepSeek R1 is forcing a shift in AI economics, emphasizing efficiency over massive capital investments. As a result, competition in the AI sector is intensifying, ushering in a “warring states era” where companies are scrambling for innovation in cost-effective models.

**Geopolitical and National Security Implications**

The success of DeepSeek R1 has intensified concerns that the U.S. is losing its technological edge to China. Policymakers are reassessing export controls on advanced chips in light of DeepSeek's ability to innovate using restricted hardware. Security concerns have also prompted the U.S. Navy to ban the use of DeepSeek R1 due to potential security and ethical risks, fueling debates over the implications of adopting foreign-developed AI systems.

**Open-Source vs Proprietary Models**

DeepSeek R1 is accelerating the democratization of AI by lowering barriers for smaller developers and researchers, fostering innovation. However, transparency concerns remain as DeepSeek has not disclosed its training data, raising ethical and bias-related questions.

**Ethical and Technical Questions**

Concerns have emerged regarding potential censorship, as some versions of DeepSeek R1 appear to align with Chinese narratives. Additionally, skepticism exists over whether DeepSeek’s reported costs and capabilities are fully accurate, with some experts questioning the factors that contributed to its success.

**Public Sentiment and the Future of AI**

Public reaction to DeepSeek-R1 has been mixed. Some view this as a “Sputnik moment,” encouraging U.S. firms to accelerate AI innovation while leveraging open-source models to stay competitive. Others see it as a wake-up call, with former President Donald Trump urging U.S. industries to adapt quickly to maintain leadership in AI development.

