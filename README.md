# Dragon Hatchling

## **Bridging the Gap Between Transformers and the Brain**

** Dragon Hatchling (BDH)** is a biologically inspired large language model architecture that connects principles of deep learning with the foundations of neuroscience. Developed by researchers at [Pathway](https://pathway.com), BDH provides a theoretical and practical framework for understanding the emergence of reasoning and generalization in artificial systems.

This repository contains the official implementation from the paper:
> *A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, M. Bartoszkiewicz.*
> [_The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain_](https://doi.org/10.48550/arXiv.2509.26507), arXiv (2025).


## Overview

BDH represents a **scale-free, locally interacting network of neurons** capable of intrinsic reasoning dynamics. BDH scales like a Transformer on performance benchmarks—yet retains full interpretability and theoretical grounding in the fine-grained dynamics of neuron interactions.

**Key properties:**

- **Scale-free network topology** mimicking biological connectivity
- **Locally interacting neuron particles** with excitatory/inhibitory dynamics
- **Hebbian working memory** based on synaptic plasticity, displaying monosemanticity
- **GPU-friendly state-space formulation** for efficient implementation
- **Interpretable activations** that are sparse and positive

BDH formalizes a bridge between **neural computation and machine-based language understanding**. It shows how **macro reasoning behavior** in large AI models emerges from **micro-level neuron dynamics**, guided by principles of graph theory and local computation.

Empirically, BDH matches **GPT-2–scale Transformers** across language and translation tasks at equivalent parameter scales (10M–1B).


***

## Architecture

<img src="figs/architecture.png" width="600"/>

***

## Relation to Transformers

<img src="figs/vocab.png" width="600"/>

BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention **emerge naturally from neuron-level interactions**, reflecting attention as seen in biological systems.

***

## Scaling Laws

<img src="figs/bdh_scaling.png" width="600"/>

BDH follows **Transformer-like scaling laws**, maintaining parameter efficiency while achieving interpretability at any scale.

***

## Latest research update: Sudoku Benchmark

Note: The Sudoku Extreme result refers to Pathway’s internal BDH implementation, not to the current open-source repository. This repository contains the implementation of the baseline variant as described in our [public paper](https://arxiv.org/abs/2509.26507) and does not reproduce the 97.4% benchmark result out of the box. See the dedicated Extreme Sudoku research blog post for additional benchmark context and the reported results.

On Sudoku Extreme, BDH reaches 97.4% accuracy across roughly 250,000 difficult puzzles, without chain-of-thought, solution backtracking, or external tool use, while leading LLMs struggle to perform on the benchmark at all.

Language is not enough for intelligence. Transformers process information token by token with limited internal state, which makes search-heavy, non-linguistic reasoning tasks like Sudoku awkward. BDH uses a larger latent reasoning space with intrinsic memory that supports learning and adaptation during use.

We believe that the future of AI will belong to systems that can reason natively across domains, that can hold multiple possibilities in a rich latent space, and that can converge on solutions without needing to verbalize every step. BDH is our answer to that challenge. It is designed to be a universal reasoning system that can speak our language without being trapped inside it. And yes, it solves Sudoku.

Read more: [Post-transformers: Sudoku Bench](https://pathway.com/research/beyond-transformers-sudoku-bench)

### Performance Comparison

| Model | Sudoku Extreme Accuracy | Relative Cost |
|------|------------------------|--------------|
| Pathway BDH | 97.4% | 10× lower, No chain-of-thought |
| Leading LLMs (O3-mini, DeepSeek R1, Claude 3.7 8K) | ~0% | High (chain-of-thought) |

*Table 1: Performance comparison on extreme Sudoku benchmarks (~250,000 difficult puzzles).*  
*Source: Pathway internal data and https://arxiv.org/pdf/2506.21734 for the Leading LLMs’ accuracy score. Pathway’s approach reflects top-1 accuracy and does not rely on chain-of-thought nor solution backtracking.*


## Installation and Training

```bash
# install dependencies
pip install -r requirements.txt

# train BDH on a toy dataset
python train.py
```

<!--For visualization and interpretability analysis, explore the example notebooks in `notebooks/`.-->



## Learn and Discuss

- Watch the *SuperDataScience podcast* [▶️ *Dragon Hatchling: The Missing Link Between Transformers and the Brain*](https://www.youtube.com/watch?v=mfV44-mtg7c) (72 min.) featuring Adrian Kosowski in conversation with Jon Krohn, unpacking BDH’s neuron-level architecture and sparse reasoning dynamics.

- Read about BDH in
[*Forbes*](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/),
[*Semafor*](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain),
[*The Turing Post*](https://www.turingpost.com/p/fod-121-300-million-to-start-a-big-promise-for-science#the-freshest-research-papers-catego),
[*Quantum Zeitgeist*](https://quantumzeitgeist.com/palo-alto-ai-firm-pathway-unveils-post-transformer-architecture-for-autonomous-ai/),
[*Golem*](https://www.golem.de/news/neue-ki-architektur-was-ist-baby-dragon-hatchling-2510-201047-2.html),
and elsewhere in the media.

- Discuss and share the BDH paper on:
[*Hugging Face Papers*](https://huggingface.co/papers/2509.26507), 
[*Alphaxiv*](https://alphaxiv.org/abs/2509.26507),
and [*EmergentMind*](https://emergentmind.com/papers/2509.26507).

## Community Projects

- [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
- [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
- [severian42/bdh](https://github.com/severian42/bdh): MLX port
- [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
- [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

## Acknowledgements
We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shapespeare dataset used in this demonstration.

BDH research stands at the intersection of **AI architecture**, **biological learning models**, and **theoretical computer science**—an effort to map the *equations of reasoning* between artificial and biological intelligence.
