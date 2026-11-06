# Dragon Hatchling (BDH): Post-Transformer AI architecture

**BDH (Dragon Hatchling)** is a **Post-Transformer AI architecture** developed by [Pathway](https://pathway.com/), a frontier AI lab. The BDH architecture makes reasoning, memory and learning part of the model itself. The [BDH explainer](https://pathway.com/research/bdh-explainer/conclusion) describes the architecture and its broader research direction.

This repository primarily includes a PyTorch implementation published with the [original BDH paper](https://arxiv.org/abs/2509.26507v1) that introduced the BDH architecture. You’ll find here a minimalistic language modeling example on the Tiny Shakespeare dataset.

**Repo’s scope:** Pathway’s research includes parametric memory, continual learning, and latent reasoning capabilities. The systems used for those results or their pretrained checkpoints are not included here. This toy example **does not** showcase how BDH models learn continually during inference or reproduce the breakthrough results published by Pathway.

[Original paper](https://arxiv.org/abs/2509.26507v1) · [BDH explainer](https://pathway.com/research/bdh-explainer/brain-inspired-ai-architecture) · [BDH-CQ paper](https://arxiv.org/abs/2608.09888v1) · [ARC task generator](https://github.com/pathwaycom/arc-task-gen) · [Installation](#installation-and-training)

## BDH-CQ: In-Context Learning with Recurrent Latent Reasoning

**August 2026.** We introduced BDH-CQ, a reasoning system built on the BDH architecture. It learns from demonstrations through recurrent memory, then works through the query in a structured, continuous latent space without generating a chain of thought.

On the public ARC-AGI-1 evaluation set, BDH-CQ, a 150-million-parameter reasoning model, achieved 29.5% pass@2 at a computed inference cost of \$0.0007 per task, breaking the cost-efficiency frontier.

Separately, early pretraining experiments from **1B to 600B parameters** showed that the BDH architecture follows scaling laws similar to those of Transformers while preserving its recurrent latent reasoning capabilities. See [Section 9.2 of the BDH-CQ paper](https://arxiv.org/html/2608.09888v1#S9.SS2).

![ARC-AGI-1 score versus computed inference cost per task, highlighting BDH-CQ at 29.5% pass@2 and $0.0007 per task.](https://raw.githubusercontent.com/pathwaycom/arc-task-gen/20b2203064b09f60f7925a191d75c11d72277f35/assets/Pathway_ARC-AGI_result_chart.png)

*ARC-AGI-1 score versus computed cost per task. Data points were collected from the official ARC Prize leaderboard on August 4, 2026. The mark at the left margin indicates the 29.5% pass@2. The point breaks through the previously reported cost and accuracy Pareto frontier and establishes a new state of the art in ARC-AGI-1 cost efficiency.*

[Read the BDH-CQ research article](https://pathway.com/research/introducing-bdh-cq) · [Read the paper](https://arxiv.org/abs/2608.09888v1)

## Overview of the Dragon Hatchling (BDH) architecture

Dragon Hatchling (BDH) is a Post-Transformer AI architecture. It represents a scale-free, locally interacting network of neurons. Its GPU formulation connects this graph view to efficient tensor computation. The graph can be recovered and inspected, and trained models exhibit interpretable synapses, sparse activity, and Transformer-like scaling.

**Key properties:**

* Scale-free network topology mimicking biological connectivity
* Locally interacting neuron particles with excitatory/inhibitory dynamics
* Hebbian working memory based on synaptic plasticity, displaying monosemanticity
* GPU-friendly state-space formulation for efficient implementation
* Interpretable activations that are sparse and positive

BDH paper: [The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain](https://arxiv.org/abs/2509.26507v1), A. Kosowski, P. Uznański, J. Chorowski, Z. Stamirowska, and M. Bartoszkiewicz (2025).

## Architecture

<img src="figs/architecture.png" width="800" alt="BDH GPU layer showing neuron activations, learned matrices, and linear attention." />

*Figure: [Architecture illustration for the original BDH model and its GPU formulation, from the September 2025 research.](figs/architecture.png)*

## Relation to Transformers

<img src="figs/vocab.png" width="800" alt="Comparison of Transformer, BDH-GPU, graph BDH, and brain models across attention mechanisms, local dynamics, and feed-forward networks." />

BDH and the Transformer share attention-inspired computation; however, BDH’s graph-based architecture makes its attention emerge naturally from neuron-level interactions, reflecting attention as seen in biological systems. The [architecture derivation](https://pathway.com/research/bdh-explainer/bdh-architecture-derivation) explains the relationship between the graph model and the tensor formulation.

**How BDH differs from the Transformer architecture**

| Dimension | BDH | Transformer |
| ----- | ----- | ----- |
| Attention | Synaptic memory with linear attention | Attention normalized with softmax |
| Key and query representation | Neuron activity in a large, sparse space | Dense keys and queries in a smaller space |
| Retrieval interpretation | Lookup interpreted through locality-sensitive hashing | Search interpreted through approximate nearest neighbors |
| Runtime memory | Synaptic state of fixed size; context has no fixed length boundary but is limited by information capacity | A token-indexed KV cache that grows as the sequence gets longer |
| Activations | Sparse vectors with non-negative entries | Dense activation vectors |
| Hardware bridge | A graph represented implicitly through low-rank tensor operations | Dense tensor operations directly |

Source: [BDH explainer, Chapter 2](https://pathway.com/research/bdh-explainer/bdh-architecture-derivation).

## Earlier research: Sudoku Extreme

In [March 2026](https://pathway.com/research/beyond-transformers-sudoku-bench), Pathway reported 97.4% accuracy on Sudoku Extreme using its internal BDH implementation, without chain of thought, solution backtracking, or external tools.

## Installation and Training

Run the following from the repository directory:

```bash
# Install dependencies
pip install -r requirements.txt

# Train the toy implementation on a toy dataset
python train.py
```

The example trains a small language model on tiny Shakespeare. It is an entry point for studying the original architecture paper.

## Learn and Discuss

* BDH: The Missing Link between the Transformer and Models of the Brain [https://arxiv.org/abs/2509.26507](https://arxiv.org/abs/2509.26507)
* BDH-CQ: In-Context Learning with Recurrent Latent Reasoning, [https://arxiv.org/abs/2608.09888](https://arxiv.org/abs/2608.09888)
* Research seminar given at MILA by our CTO and BDH co-inventor, Jan Chorowski, a co-author of Yoshua Bengio and Nobel Prize winner Geoff Hinton [https://pathway.com/blog/mila-bdh](https://pathway.com/blog/mila-bdh)
* Read the three part BDH explainer: [motivation](https://pathway.com/research/bdh-explainer/brain-inspired-ai-architecture), [architecture](https://pathway.com/research/bdh-explainer/bdh-architecture-derivation), and [interpretability and scaling](https://pathway.com/research/bdh-explainer/bdh-interpretability-scaling).
* Watch the [SuperDataScience conversation with Adrian Kosowski and Jon Krohn](https://www.youtube.com/watch?v=mfV44-mtg7c).
* Read coverage in [Forbes](https://www.forbes.com/sites/victordey/2025/10/08/can-ai-learn-and-evolve-like-a-brain-pathways-bold-research-thinks-so/), [Semafor](https://www.semafor.com/article/10/01/2025/new-ai-research-claims-to-be-getting-closer-to-modeling-human-brain), [The Turing Post](https://www.turingpost.com/p/fod-121-300-million-to-start-a-big-promise-for-science#the-freshest-research-papers-catego), [Quantum Zeitgeist](https://quantumzeitgeist.com/palo-alto-ai-firm-pathway-unveils-post-transformer-architecture-for-autonomous-ai/), and [Golem](https://www.golem.de/news/neue-ki-architektur-was-ist-baby-dragon-hatchling-2510-201047-2.html).
* Discuss the original paper on [Hugging Face Papers](https://huggingface.co/papers/2509.26507), [Alphaxiv](https://alphaxiv.org/abs/2509.26507), and [EmergentMind](https://emergentmind.com/papers/2509.26507).

## Community Projects

* [adamskrodzki/bdh](https://github.com/adamskrodzki/bdh): dynamic vocabulary, stateful attention
* [mosure/burn_dragon_hatchling](https://github.com/mosure/burn_dragon_hatchling): Burn port
* [severian42/bdh](https://github.com/severian42/bdh): MLX port
* [Git-Faisal/bdh](https://github.com/Git-Faisal/bdh)
* [GrahLnn/bdh](https://github.com/GrahLnn/bdh)

## Acknowledgements

We thank Andrej Karpathy for the [nanoGPT](https://github.com/karpathy/nanoGPT/) code and the tiny Shakespeare dataset used in this demonstration.

## License

The code in this repository is available under the [MIT license](LICENSE.md).
