# Awesome Mobile LLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of LLMs and related studies targeted at mobile and embedded hardware

Last update: 10th April 2024

If your publication/work is not included - and you think it should - please open an issue or reach out directly to @stevelaskaridis.

Let's try to make this list as useful as possible to researchers, engineers and practitioners all around the world.

## Contents

- [Mobile-First LLMs](#Mobile-First-LLMs)
- [Infrastructure / Deployment of LLMs on Device](#Infrastructure-/-Deployment-of-LLMs-on-Device)
- [Benchmarking LLMs on Device](#Benchmarking-LLMs-on-Device)
- [Applications](#Applications)
- [Multimodal LLMs](#Multimodal-LLMs)
- [Surveys on Efficient LLMs](#Surveys-on-Efficient-LLMs)
- [Training LLMs on Device](#Training-LLMs-on-Device)
- [Mobile-Related Use-Cases](#Mobile-Related-Use-Cases)
- [Related Awesome Repositories](#Related-Awesome-Repositories)


## Mobile-First LLMs

The following Table shows sub-3B models designed for on-device deployments, sorted by year.

| Name   | Year | Sizes               | Primary Group/Affiliation                               | Publication                                 | Code Repository                                  | HF Repository                                             |
| ---    | --- | ---                | ---                                             | ---                                           | ---                                              | ---                                                       |
| OpenELM | 2024 | 270M, 450M, 1.08B, 3.04B | Apple | [paper](https://arxiv.org/abs/2404.14619)  | [code](https://github.com/apple/corenet) | [huggingface](https://huggingface.co/apple/OpenELM) |
| Phi-3 | 2024 | 3.8B | Microsoft | [whitepaper](https://arxiv.org/abs/2404.14219) |  - | [huggingface](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) |
| OLMo | 2024 | 1B, ... | AllenAI | [paper](https://arxiv.org/abs/2402.00838) | [code](https://github.com/allenai/OLMo) | [huggingface](https://huggingface.co/allenai/OLMo-7B) |
| Mobile LLMs | 2024 | 125M, 250M | Meta                                            | [paper](https://arxiv.org/abs/2402.14905)   | -                                                  | -                                                           |
| Gemma | 2024 | 2B, ...             | Google                                          | [website](https://ai.google.dev/gemma)      | [code](https://github.com/google-deepmind/gemma), [gemma.cpp](https://github.com/google/gemma.cpp) | [huggingface](https://huggingface.co/google/gemma-2b) |
| MobiLlama | 2024 | 0.5B, 1B        | MBZUAI | [paper](https://arxiv.org/abs/2402.16840)   | [code](https://github.com/mbzuai-oryx/MobiLlama) | [huggingface](https://huggingface.co/MBZUAI/MobiLlama-1B) |
| TinyLlama | 2024 | 1.1B            | Singapore University of Technology and Design   | [paper](https://arxiv.org/abs/2401.02385)   | [code](https://github.com/jzhang38/TinyLlama)    | [huggingface](https://huggingface.co/TinyLlama)           |
| Gemini-Nano | 2024 | 1.8B, 3.25B  | Google                                          | [paper](https://arxiv.org/abs/2312.11805)      | - | - |
| OpenLM | 2023 | 11M, 25M, 87M, 160M, 411M, 830M, 1B, 3B, ... | OpenLM team | - | [code](https://github.com/mlfoundations/open_lm/) | [huggingface](https://huggingface.co/mlfoundations/open_lm_1B) |
| Phi-2  | 2023 | 2.7B               | Microsoft                                       | [website](https://www.microsoft.com/en-us/research/blog/phi-2-the-surprising-power-of-small-language-models/)      | - | [huggingface](https://huggingface.co/microsoft/phi-2) |
| Phi-1.5 | 2023 | 1.3B             | Microsoft                                       | [paper](https://arxiv.org/abs/2309.05463)   | - | [huggingface](https://huggingface.co/microsoft/phi-1_5) |
| Phi-1  | 2023| 1.3B               | Microsoft                                       | [paper](https://arxiv.org/abs/2306.11644)   | - | [huggingface](https://huggingface.co/microsoft/phi-1) |
| RWKV | 2023 |  169M, 430M, 1.5B, 3B, ... | EleutherAI | [paper](https://arxiv.org/abs/2305.13048) | [code](https://github.com/BlinkDL/RWKV-LM) | [huggingface](https://huggingface.co/RWKV) |
| Cerebras-GPT | 2023 | 111M, 256M, 590M, 1.3B, 2.7B ... | Cerebras | [paper](https://arxiv.org/abs/2304.03208) | [code](https://github.com/Cerebras/modelzoo) | [huggingface](https://huggingface.co/cerebras) |
| OPT | 2022 | 125M, 350M, 1.3B, 2.7B, ...          | Meta                                            | [paper](https://arxiv.org/abs/2205.01068)   | [code](https://github.com/facebookresearch/metaseq) | [huggingface](https://huggingface.co/facebook/opt-350m) |
| LaMini-LM | 2023 | 61M, 77M, 111M, 124M, 223M, 248M, 256M, 590M, 774M, 738M, 783M, 1.3B, 1.5B, ... | MBZUAI | [paper](https://arxiv.org/pdf/2304.14402.pdf) | [code](https://arxiv.org/pdf/2304.14402.pdf) | [huggingface](https://huggingface.co/MBZUAI/LaMini-T5-61M) |
| Pythia | 2023 | 70M, 160M, 410M, 1B, 1.4B, 2.8B, ...         | EleutherAI                                      | [paper](https://arxiv.org/abs/2304.01373)   | [code](https://github.com/EleutherAI/pythia)     | [huggingface](https://huggingface.co/EleutherAI/pythia-70m-deduped) |
| Galactica | 2022 | 125M, 1.3B, ... | Meta | [paper](https://arxiv.org/abs/2211.09085) | [code](https://github.com/facebookresearch/metaseq/) | [huggingface](https://huggingface.co/facebook/galactica-125m) |
| BLOOM | 2022 | 560M, 1.1B, 1.7B, 3B, ... | BigScience | [paper](https://arxiv.org/abs/2211.05100) | [code](https://github.com/bigscience-workshop/bigscience/tree/master) | [huggingface](https://huggingface.co/docs/transformers/en/model_doc/bloom) |
| XGLM | 2021 | 564M, 1.7B, 2.9B, ... | Meta | [paper](https://arxiv.org/abs/2112.10668) | [code](https://github.com/facebookresearch/fairseq/tree/main/examples/xglm) | [huggingface](https://huggingface.co/facebook/xglm-564M) |
| GPT-Neo| 2021 | 125M, 350M, 1.3B, 2.7B | EleutherAI                                  | -  | [code](https://github.com/EleutherAI/gpt-neo), [gpt-neox](https://github.com/EleutherAI/gpt-neox/)    | [huggingface](https://huggingface.co/EleutherAI/gpt-neo-125m) |
| MobileBERT | 2020 | 15.1M, 25.3M | CMU, Google | [paper](https://arxiv.org/abs/2004.02984) | [code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mobilebert/modeling_mobilebert.py) | [huggingface](https://huggingface.co/google/mobilebert-uncased) |
| BART | 2019 | 140M, 400M           | Meta                                            | [paper](https://arxiv.org/abs/1910.13461)   | [code](https://github.com/facebookresearch/fairseq/tree/main/examples/bart) | [huggingface](https://huggingface.co/facebook/bart-base) |
| DistilBERT | 2019 | 66M | HuggingFace | [paper](https://arxiv.org/pdf/1910.01108.pdf) | [code](https://github.com/huggingface/transformers/tree/main/examples/research_projects/distillation) | [huggingface](https://huggingface.co/distilbert/distilbert-base-uncased) |
| T5 | 2019 | 60M, 220M, 770M, 3B, ... |  Google | [paper](https://arxiv.org/abs/1910.10683) | [code](https://github.com/google-research/text-to-text-transfer-transformer) | [huggingface](https://huggingface.co/google-t5/t5-small) |
| TinyBERT | 2019 | 14.5M | Huawei | [paper](https://arxiv.org/abs/1909.10351) | [code](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) | [huggingface](https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D) |
| Megatron-LM | 2019 | 336M, 1.3B, ... | Nvidia | [paper](https://arxiv.org/abs/1909.08053) | [code](https://github.com/NVIDIA/Megatron-LM) | - |


## Infrastructure / Deployment of LLMs on Device

This section showcases frameworks and contributions for supporting LLM inference on mobile and edge devices.

### Deployment Frameworks

- [llama.cpp](https://github.com/ggerganov/llama.cpp)
    - [LLMFarm](https://github.com/guinmoon/LLMFarm): iOS frontend for llama.cpp
    - [Sherpa](https://github.com/Bip-Rep/sherpa): Android frontend for llama.cpp
    - [dusty-nv's llama.cpp](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/llama_cpp): Containers for Jetson deployment of llama.cpp
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
    - [Android App](https://llm.mlc.ai/#android): MLC Android app
    - [iOS App](https://llm.mlc.ai/#ios): MLC iOS app
    - [dusty-nv's MLC](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/mlc): Containers for Jetson deployment of MLC
- [Google MediaPipe](https://github.com/google/mediapipe)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [Alibaba MNN](https://github.com/alibaba/MNN)
- [llama2.c](https://github.com/karpathy/llama2.c) (More educational, see [here](https://github.com/Manuel030/llama2.c-android) for android port)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [TinyChatEngine](https://github.com/mit-hanlab/TinyChatEngine) (Targeted at Nvidia, Apple M1 and RPi)

### Papers

#### 2024

- **[MobiCom'24]** Mobile Foundation Model as Firmware ([paper](https://xumengwei.github.io/files/MobiCom24-MobileFM.pdf), [code](https://github.com/UbiquitousLearning/MobileFM))
- Merino: Entropy-driven Design for Generative Language Models on IoT Devicess ([paper](https://arxiv.org/abs/2403.07921))
- LLM as a System Service on Mobile Devices ([paper](https://arxiv.org/abs/2403.11805))

#### 2023

- LLMCad: Fast and Scalable On-device Large Language Model Inference ([paper](https://arxiv.org/abs/2309.04255))
- EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models ([paper](https://arxiv.org/abs/2308.14352))

#### 2022

- The Future of Consumer Edge-AI Computing ([paper](https://arxiv.org/abs/2210.10514), [talk](https://www.youtube.com/watch?v=WyKxGKy_rnk))

## Benchmarking LLMs on Device

This section focuses on measurements and benchmarking efforts for assessing LLM performance when deployed on device.

### Papers

#### 2024

- MELTing point: Mobile Evaluation of Language Transformers ([paper](https://arxiv.org/abs/2403.12844))

## Applications

### Papers

#### 2024

- Octopus v2: On-device language model for super agent ([paper](https://arxiv.org/abs/2404.01744))

#### 2023

- Towards an On-device Agent for Text Rewriting ([paper](https://arxiv.org/abs/2308.11807))

## Multimodal LLMs

This section refers to multimodal LLMs, which integrate vision or other modalities in their tasks.

### Papers

#### 2024

- TinyLLaVA: A Framework of Small-scale Large Multimodal Models ([paper](https://arxiv.org/abs/2402.14289), [code](https://github.com/DLCV-BUAA/TinyLLaVABench))
- MobileVLM V2: Faster and Stronger Baseline for Vision Language Model ([paper](https://arxiv.org/abs/2402.03766), [code](https://github.com/Meituan-AutoML/MobileVLM))

#### 2023

- MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile Devices ([paper](https://arxiv.org/abs/2312.16886), [code](https://github.com/Meituan-AutoML/MobileVLM))


## Surveys on Efficient LLMs

This section includes survey papers on LLM efficiency, a topic very much related to deploying in constrained devices.

### Papers

#### 2024

- A Survey of Resource-efficient LLM and Multimodal Foundation Models ([paper](https://arxiv.org/pdf/2401.08092.pdf))

#### 2023

- Efficient Large Language Models: A Survey ([paper](https://arxiv.org/abs/2312.03863), [code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey))
- Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems ([paper](https://arxiv.org/abs/2312.15234))
- A Survey on Model Compression for Large Language Models ([paper](https://arxiv.org/abs/2308.07633))


## Training LLMs on Device

This section refers to papers attempting to train/fine-tune LLMs on device, in a standalone or federated manner.

### Papers

#### 2023

- **[MobiCom'23]** Federated Few-Shot Learning for Mobile NLP ([paper](https://arxiv.org/abs/2212.05974), [code](https://github.com/UbiquitousLearning/FeS))
- FwdLLM: Efficient FedLLM using Forward Gradient ([paper](https://arxiv.org/abs/2308.13894), [code](https://github.com/UbiquitousLearning/FwdLLM))
- **[Electronics'24]** Forward Learning of Large Language Models by Consumer Devices ([paper](https://www.mdpi.com/2079-9292/13/2/402))
- Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly ([paper](https://arxiv.org/pdf/2310.03150.pdf))
- Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes ([paper](https://arxiv.org/abs/2312.06353), [code](https://github.com/alibaba/FederatedScope/tree/FedKSeed))

## Mobile-Related Use-cases

This section includes paper that are mobile-related, but not necessarily run on device.

### Papers

#### 2024

- Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs ([paper](https://arxiv.org/abs/2404.05719))
- Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception ([paper](https://arxiv.org/abs/2401.16158), [code](https://github.com/X-PLUG/MobileAgent))

#### 2023

- [NeurIPS'23] AndroidInTheWild: A Large-Scale Dataset For Android Device Control ([paper](https://arxiv.org/abs/2307.10088), [code](https://github.com/google-research/google-research/tree/master/android_in_the_wild))
- GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation ([paper](https://arxiv.org/abs/2311.07562), [code](https://github.com/zzxslp/MM-Navigator))

#### Older

- [ACL'20] Mapping Natural Language Instructions to Mobile UI Action Sequences ([paper](https://arxiv.org/abs/2005.03776))

## Related Awesome Repositories

If you want to read more about related topics, here are some tangential awesome repositories to visit:

* [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) on Large Language Models
* [KennethanCeyer/awesome-llm](https://github.com/KennethanCeyer/awesome-llm) on  Large Language Models
* [HuangOwen/Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) on Large Language Model Compression
* [csarron/awesome-emdl](https://github.com/csarron/awesome-emdl) on Embedded and Mobile Deep Learning


## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.
