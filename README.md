# Awesome Mobile LLMs [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of LLMs and related studies targeted at mobile and embedded hardware

Last update: 27th July 2025

If your publication/work is not included - and you think it should - please open an issue or reach out directly to [@stevelaskaridis](https://github.com/stevelaskaridis).

Let's try to make this list as useful as possible to researchers, engineers and practitioners all around the world.

## Contents

- [Mobile-First LLMs](#Mobile-First-LLMs)
- [Infrastructure / Deployment of LLMs on Device](#Infrastructure-/-Deployment-of-LLMs-on-Device)
- [Benchmarking LLMs on Device](#Benchmarking-LLMs-on-Device)
- [Mobile-Specific Optimisations](#Mobile-Specific-Optimisations)
- [Applications](#Applications)
- [Multimodal LLMs](#Multimodal-LLMs)
- [Surveys on Efficient LLMs](#Surveys-on-Efficient-LLMs)
- [Training LLMs on Device](#Training-LLMs-on-Device)
- [Mobile-Related Use-Cases](#Mobile-Related-Use-Cases)
- [Benchmarks](#Benchmarks)
- [Leaderboards](#Leaderboards)
- [Industry Announcements](#Industry-Announcements)
- [Books/Courses](#Books-and-Courses)
- [Related Organized Workshops](#Related-Organized-Workshops)
- [Related Awesome Repositories](#Related-Awesome-Repositories)


## Mobile-First LLMs

The following Table shows sub-3B models designed for on-device deployments, sorted by year.

| Name   | Year | Sizes               | Primary Group/Affiliation                               | Publication                                 | Code Repository                                  | HF Repository                                             |
| ---    | --- | ---                | ---                                             | ---                                           | ---                                              | ---                                                       |
| SmolLM3 | 2025 | 3B | HuggingFace | [blog](https://huggingface.co/blog/smollm3) | [code](https://github.com/huggingface/transformers/tree/main/src/transformers/models/smollm3) | [huggingface](https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base) |
| Qwen-3 | 2025 | 0.6B, 1.7B, ... | Qwen Team | [paper](https://arxiv.org/abs/2505.09388) | [code](https://github.com/QwenLM/Qwen3) | [huggingface](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Pareto-Q | 2025 | 125M, 350M, 600M, 1B, 1.5B, 3B | Meta | [paper](https://arxiv.org/abs/2502.02631) | [code](https://github.com/facebookresearch/ParetoQ) | [huggingface](https://huggingface.co/facebook/MobileLLM-ParetoQ-1.5B-1.58-bit) |
| BlueLM-V | 2024 | 2.7B | CUHK, Vivo AI Lab | [paper](https://arxiv.org/abs/2411.10640) | [code](https://github.com/vivo-ai-lab/BlueLM) | - |
| PhoneLM | 2024 | 0.5B, 1.5B | BUPT | [paper](https://arxiv.org/abs/2411.05046) | [code](https://github.com/UbiquitousLearning/PhoneLM) | [huggingface](https://huggingface.co/mllmTeam/PhoneLM-0.5B) |
| AMD-Llama-135m | 2024 | 135M | AMD | [blog](https://community.amd.com/t5/ai/amd-unveils-its-first-small-language-model-amd-135m/ba-p/711368) | [code](https://github.com/AMD-AIG-AIMA/AMD-LLM) | [huggingface](https://huggingface.co/amd/AMD-Llama-135m) |
| SmolLM2 | 2024 | 135M, 360M, 1.7B | Huggingface | - | [code](https://github.com/huggingface/smollm) |[huggingface](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9) |
| Ministral | 2024 | 3B, ... | Mistral | [blog](https://mistral.ai/news/ministraux/) | - | [huggingface](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) |
| Llama 3.2 | 2024 | 1B, 3B | Meta | [blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/) | [code](https://github.com/meta-llama/llama-models) | [huggingface](https://huggingface.co/meta-llama/Llama-3.2-1B) |
| OLMoE | 2024 | 7B (1B active) | AllenAI | [paper](https://arxiv.org/abs/2409.02060) | [code](https://github.com/allenai/OLMoE) | [huggingface](https://hf.co/allenai/OLMoE-1B-7B-0924) |
| Spectra | 2024 | 99M - 3.9B | NolanoAI | [paper](https://arxiv.org/abs/2407.12327) | [code](https://github.com/NolanoOrg/SpectraSuite) | [huggingface](https://huggingface.co/collections/SpectraSuite/trilms-unpacked-668d5f62afe0f4036925b1d2) |
| Gemma 2 | 2024 | 2B, ... | Google | [paper](https://arxiv.org/abs/2408.00118) [blog](https://developers.googleblog.com/en/smaller-safer-more-transparent-advancing-responsible-ai-with-gemma/) | [code](https://github.com/google/gemma_pytorch) | [huggingface](https://huggingface.co/google/gemma-2-2b-it) |
| Apple Intelligence Foundation LMs | 2024 | 3B | Apple | [paper](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models) | - | - |
| SmolLM | 2024 | 135M, 360M, 1.7B | Huggingface | [blog](https://huggingface.co/blog/smollm) | - | [huggingface](https://huggingface.co/HuggingFaceTB/SmolLM-135M) |
| Fox | 2024 | 1.6B | TensorOpera | [blog](https://blog.tensoropera.ai/tensoropera-unveils-fox-foundation-model-a-pioneering-open-source-slm-leading-the-way-against-tech-giants/) | - | [huggingface](https://huggingface.co/tensoropera/Fox-1-1.6B) |
| Qwen2 | 2024 | 500M, 1.5B, ... | Qwen Team | [paper](https://arxiv.org/abs/2309.16609) | [code](https://github.com/QwenLM/Qwen2) | [huggingface](https://huggingface.co/Qwen/Qwen2-0.5B) |
| OpenELM | 2024 | 270M, 450M, 1.08B, 3.04B | Apple | [paper](https://arxiv.org/abs/2404.14619)  | [code](https://github.com/apple/corenet) | [huggingface](https://huggingface.co/apple/OpenELM) |
| DCLM | 2024 | 400M, 1B, ... | Univerisy of Washington, Apple, Toyota Research Institute, ... | [paper](https://arxiv.org/abs/2406.11794) | [code](https://github.com/mlfoundations/dclm) | [huggingface](https://huggingface.co/TRI-ML/DCLM-1B) |
| Phi-3 | 2024 | 3.8B | Microsoft | [whitepaper](https://arxiv.org/abs/2404.14219) | [code](https://github.com/microsoft/Phi-3CookBook) | [huggingface](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) |
| BitNet-b1.58 | 2024 | 1.3B, 3B, ... | Microsoft | [paper](https://arxiv.org/abs/2402.17764) | [code](https://github.com/microsoft/BitNet) | [huggingface](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T) |
| OLMo | 2024 | 1B, ... | AllenAI | [paper](https://arxiv.org/abs/2402.00838) | [code](https://github.com/allenai/OLMo) | [huggingface](https://huggingface.co/allenai/OLMo-7B) |
| Mobile LLMs | 2024 | 125M, 250M | Meta                                      | [paper](https://arxiv.org/abs/2402.14905)   | [code](https://github.com/facebookresearch/MobileLLM)                           | -                                                           |
| Gemma | 2024 | 2B, ...             | Google                                          | [paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf), [website](https://ai.google.dev/gemma)      | [code](https://github.com/google-deepmind/gemma), [gemma.cpp](https://github.com/google/gemma.cpp) | [huggingface](https://huggingface.co/google/gemma-2b) |
| MobiLlama | 2024 | 0.5B, 1B        | MBZUAI | [paper](https://arxiv.org/abs/2402.16840)   | [code](https://github.com/mbzuai-oryx/MobiLlama) | [huggingface](https://huggingface.co/MBZUAI/MobiLlama-1B) |
| Stable LM 2 (Zephyr) | 2024 | 1.6B | Stability.ai | [paper](https://drive.google.com/file/d/1JYJHszhS8EFChTbNAf8xmqhKjogWRrQF/view) | - | [huggingface](https://huggingface.co/stabilityai/stablelm-2-1_6b) |
| TinyLlama | 2024 | 1.1B            | Singapore University of Technology and Design   | [paper](https://arxiv.org/abs/2401.02385)   | [code](https://github.com/jzhang38/TinyLlama)    | [huggingface](https://huggingface.co/TinyLlama)           |
| Gemini-Nano | 2024 | 1.8B, 3.25B  | Google                                          | [paper](https://arxiv.org/abs/2312.11805)      | - | - |
| Stable LM (Zephyr) | 2023 | 3B           | Stability | [blog](https://stability.ai/news/stablelm-zephyr-3b-stability-llm) | [code](https://github.com/Stability-AI/StableLM) | [huggingface](https://huggingface.co/stabilityai/stablelm-zephyr-3b) |
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

- [llama.cpp](https://github.com/ggerganov/llama.cpp): Inference of Meta's LLaMA model (and others) in pure C/C++. Supports various platforms and builds on top of ggml (now gguf format).
    - [LLMFarm](https://github.com/guinmoon/LLMFarm): iOS frontend for llama.cpp
    - [LLM.swift](https://github.com/eastriverlee/LLM.swift): iOS frontend for llama.cpp
    - [Sherpa](https://github.com/Bip-Rep/sherpa): Android frontend for llama.cpp
    - [iAkashPaul/Portal](https://github.com/iakashpaul/portal): Wraps the example android app with tweaked UI, configs & additional model support
    - [dusty-nv's llama.cpp](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/llama_cpp): Containers for Jetson deployment of llama.cpp
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm): MLC LLM is a machine learning compiler and high-performance deployment engine for large language models. Supports various platforms and build on top of TVM.
    - [Android App](https://llm.mlc.ai/#android): MLC Android app
    - [iOS App](https://llm.mlc.ai/#ios): MLC iOS app
    - [dusty-nv's MLC](https://github.com/dusty-nv/jetson-containers/tree/master/packages/llm/mlc): Containers for Jetson deployment of MLC
- [PyTorch ExecuTorch](https://github.com/pytorch/executorch): Solution for enabling on-device inference capabilities across mobile and edge devices including wearables, embedded devices and microcontrollers.
    - [TorchChat](https://github.com/pytorch/torchchat): Codebase showcasing the ability to run large language models (LLMs) seamlessly across iOS and Android
- [Google MediaPipe](https://github.com/google/mediapipe): A suite of libraries and tools for you to quickly apply artificial intelligence (AI) and machine learning (ML) techniques in your applications. Support Android, iOS, Python and Web.
    - [GoogleAI-Edge Gallery](https://github.com/google-ai-edge/gallery): Experimental app that puts the power of cutting-edge Generative AI models directly into your hands, running entirely on your Android and iOS devices.
- [Apple MLX](https://github.com/ml-explore/mlx): MLX is an array framework for machine learning research on Apple silicon, brought to you by Apple machine learning research. Builds upon lazy evaluation and unified memory architecture.
    - [MLX Swift](https://github.com/ml-explore/mlx-swift): Swift API for MLX.
- [HF Swift Transformers](https://github.com/huggingface/swift-transformers): Swift Package to implement a transformers-like API in Swift
- [Alibaba MNN](https://github.com/alibaba/MNN): MNN supports inference and training of deep learning models and for inference and training on-device.
- [llama2.c](https://github.com/karpathy/llama2.c) (More educational, see [here](https://github.com/Manuel030/llama2.c-android) for android port)
- [tinygrad](https://github.com/tinygrad/tinygrad): Simple neural network framework from tinycorp and [@geohot](https://github.com/geohot)
- [TinyChatEngine](https://github.com/mit-han-lab/TinyChatEngine): Targeted at Nvidia, Apple M1 and RPi, from Song Han's (MIT) [group](https://hanlab.mit.edu/team).
- [Llama Stack](https://github.com/meta-llama/llama-stack) ([swift](https://github.com/meta-llama/llama-stack-client-swift), [kotlin](https://github.com/meta-llama/llama-stack-client-kotlin)): These libraries are a set of SDKs that provide a simple and effective way to integrate AI capabilities into your iOS/Android app, whether it is local (on-device) or remote inference.
- [OLMoE.Swift](https://github.com/allenai/OLMoE.swift): Ai2 OLMoE is an AI chatbot powered by the OLMoE model. Unlike cloud-based AI assistants, OLMoE runs entirely on your device, ensuring complete privacy and offline accessibility—even in Flight Mode.
- [HuggingSnap](https://github.com/huggingface/HuggingSnap): HuggingSnap is an iOS app that lets users quickly learn more about the places and objects around them. HuggingSnap runs SmolVLM2, a compact open multimodal model that accepts arbitrary sequences of image, videos, and text inputs to produce text outputs.
- [Flower Intelligence](https://flower.ai/docs/intelligence/): Flower Intelligence is a cross-platform inference library that lets users seamlessly interact with Large-Language Models both locally and remotely in a secure and private way. The library was created by the Flower Labs team. It supports TypeScript, JavaScript and Swift backends.


### Papers

#### 2025

- Apple Intelligence Foundation Language Models: Tech Report 2025 ([paper](https://arxiv.org/abs/2507.13575))
- **[ACM Queue]** Generative AI at the Edge: Challenges and Opportunities: The next phase in AI deployment ([paper](https://dl.acm.org/doi/abs/10.1145/3733702))

#### 2024

- PowerInfer-2: Fast Large Language Model Inference on a Smartphone ([paper](https://arxiv.org/abs/2406.06282), [code](https://github.com/SJTU-IPADS/PowerInfer))
- **[MobiCom'24]** Mobile Foundation Model as Firmware ([paper](https://xumengwei.github.io/files/MobiCom24-MobileFM.pdf), [code](https://github.com/UbiquitousLearning/MobileFM))
- Merino: Entropy-driven Design for Generative Language Models on IoT Devicess ([paper](https://arxiv.org/abs/2403.07921))
- LLM as a System Service on Mobile Devices ([paper](https://arxiv.org/abs/2403.11805))

#### 2023

- LLMCad: Fast and Scalable On-device Large Language Model Inference ([paper](https://arxiv.org/abs/2309.04255))
- EdgeMoE: Fast On-Device Inference of MoE-based Large Language Models ([paper](https://arxiv.org/abs/2308.14352))

#### 2022

- **[IEEE Pervasive Computing]** The Future of Consumer Edge-AI Computing ([paper](https://arxiv.org/abs/2210.10514), [talk](https://www.youtube.com/watch?v=WyKxGKy_rnk))

## Benchmarking LLMs on Device

This section focuses on measurements and benchmarking efforts for assessing LLM performance when deployed on device.

### Papers

#### 2025

- **[ICLR'25]** PalmBench: A Comprehensive Benchmark of Compressed Large Language Models on Mobile Platforms ([paper](https://arxiv.org/abs/2410.05315))

#### 2024

- Large Language Model Performance Benchmarking on Mobile Platforms: A Thorough Evaluation ([paper](https://arxiv.org/abs/2410.03613))
- **[EdgeFM @ MobiSys'24]** Large Language Models on Mobile Devices: Measurements, Analysis, and Insights ([paper](https://dl.acm.org/doi/abs/10.1145/3662006.3662059?casa_token=lSWawSGkqzUAAAAA:QhHsJqnEw4i9v8dCGMtelbulm1PqwfbFW_28x4c64eTjuz4BKA76ag6s0NsnCZPm02UdMF68hd6F))
- MobileAIBench: Benchmarking LLMs and LMMs for On-Device Use Cases ([paper](https://arxiv.org/abs/2406.10290))
- **[MobiCom'24]** MELTing point: Mobile Evaluation of Language Transformers ([paper](https://arxiv.org/abs/2403.12844), [talk](https://www.youtube.com/watch?feature=shared&t=326&v=sohvvDFT3DU), [code](https://github.com/brave-experiments/MELT-public))

## Mobile-Specific Optimisations

This section focuses on techniques and optimisations that target mobile-specific deployment.

### Papers

#### 2025

- **[CVPR'25 EDGE Workshop]** Scaling On-Device GPU Inference for Large Generative Models ([paper](https://arxiv.org/abs/2505.00232))
- ROMA: a Read-Only-Memory-based Accelerator for QLoRA-based On-Device LLM ([paper](https://arxiv.org/abs/2502.20421))
- **[ASPLOS'25]** Fast On-device LLM Inference with NPUs ([paper](https://arxiv.org/abs/2407.05858), [code](https://github.com/UbiquitousLearning/mllm))

#### 2024

- Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference ([paper](https://arxiv.org/abs/2412.00099))
- PhoneLM: An Efficient and Capable Small Language Model Family through Principled Pre-training ([paper](https://arxiv.org/abs/2411.05046), [code](https://github.com/UbiquitousLearning/PhoneLM))
- MobileQuant: Mobile-friendly Quantization for On-device Language Models ([paper](https://arxiv.org/abs/2408.13933), [code](https://github.com/saic-fi/MobileQuant))
- Gemma 2: Improving Open Language Models at a Practical Size ([paper](https://arxiv.org/abs/2408.00118), [code](https://github.com/google/gemma_pytorch))
- Apple Intelligence Foundation Language Models ([paper](https://arxiv.org/abs/2407.21075))
- Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone ([paper](https://arxiv.org/abs/2404.14219), [code](https://github.com/microsoft/Phi-3CookBook))
- Gemma: Open Models Based on Gemini Research and Technology ([paper](https://storage.googleapis.com/deepmind-media/gemma/gemma-report.pdf), [code](https://github.com/google/gemma_pytorch))
- MobiLlama: Towards Accurate and Lightweight Fully Transparent GPT ([paper](https://arxiv.org/abs/2402.16840), [code](https://github.com/mbzuai-oryx/MobiLlama))
- **[ICML'24]** MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases ([paper](https://arxiv.org/abs/2402.14905), [code](https://github.com/facebookresearch/MobileLLM))
- **[ICML'24]** Rethinking Optimization and Architecture for Tiny Language Models ([paper](https://arxiv.org/abs/2402.02791), [code](https://github.com/YuchuanTian/RethinkTinyLM))
- TinyLlama: An Open-Source Small Language Model ([paper](https://arxiv.org/abs/2401.02385), [code](https://github.com/jzhang38/TinyLlama))

## Applications

### Papers

#### 2024

- Octopus v3: Technical Report for On-device Sub-billion Multimodal AI Agent ([paper](https://arxiv.org/abs/2404.11459))
- Octopus v2: On-device language model for super agent ([paper](https://arxiv.org/abs/2404.01744))

#### 2023

- Towards an On-device Agent for Text Rewriting ([paper](https://arxiv.org/abs/2308.11807))

## Multimodal LLMs

This section refers to multimodal LLMs, which integrate vision or other modalities in their tasks.

### Papers

#### 2024

- **[CVPR 2024]** MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training ([paper](https://openaccess.thecvf.com/content/CVPR2024/html/Vasu_MobileCLIP_Fast_Image-Text_Models_through_Multi-Modal_Reinforced_Training_CVPR_2024_paper.html))
- TinyLLaVA: A Framework of Small-scale Large Multimodal Models ([paper](https://arxiv.org/abs/2402.14289), [code](https://github.com/DLCV-BUAA/TinyLLaVABench))
- MobileVLM V2: Faster and Stronger Baseline for Vision Language Model ([paper](https://arxiv.org/abs/2402.03766), [code](https://github.com/Meituan-AutoML/MobileVLM))

#### 2023

- MobileVLM : A Fast, Strong and Open Vision Language Assistant for Mobile Devices ([paper](https://arxiv.org/abs/2312.16886), [code](https://github.com/Meituan-AutoML/MobileVLM))


## Surveys on Efficient LLMs

This section includes survey papers on LLM efficiency, a topic very much related to deploying in constrained devices.

### Papers

#### 2025

- GenAI at the Edge: Comprehensive Survey on Empowering Edge Devices ([paper](https://arxiv.org/abs/2502.15816))
- Small Language Models (SLMs) Can Still Pack a Punch: A survey ([paper](https://arxiv.org/abs/2501.05465))

#### 2024

- A Comprehensive Survey of Small Language Models in the Era of Large Language Models: Techniques, Enhancements, Applications, Collaboration with LLMs, and Trustworthiness ([paper](https://arxiv.org/abs/2411.03350))
- Small Language Models: Survey, Measurements, and Insights ([paper](https://arxiv.org/abs/2409.15790))
- On-Device Language Models: A Comprehensive Review ([paper](https://arxiv.org/abs/2409.00088))
- A Survey of Resource-efficient LLM and Multimodal Foundation Models ([paper](https://arxiv.org/pdf/2401.08092.pdf))

#### 2023

- Efficient Large Language Models: A Survey ([paper](https://arxiv.org/abs/2312.03863), [code](https://github.com/AIoT-MLSys-Lab/Efficient-LLMs-Survey))
- Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems ([paper](https://arxiv.org/abs/2312.15234))
- A Survey on Model Compression for Large Language Models ([paper](https://arxiv.org/abs/2308.07633))


## Training LLMs on Device

This section refers to papers attempting to train/fine-tune LLMs on device, in a standalone or federated manner.

### Papers

### 2025

- Computational Bottlenecks of Training Small-scale Large Language Models [paper](https://arxiv.org/abs/2410.19456)
- **[ICML'25]**On-device collaborative language modeling via a mixture of generalists and specialists ([paper](https://arxiv.org/abs/2409.13931))
- MobiLLM: Enabling LLM Fine-Tuning on the Mobile Device via Server Assisted Side Tuning ([paper](https://arxiv.org/abs/2502.20421))

### 2024

- **[Privacy in Natural Language Processing @ ACL'24]** PocketLLM: Enabling On-Device Fine-Tuning for Personalized LLMs ([paper](https://aclanthology.org/2024.privatenlp-1.10.pdf))

#### 2023

- **[MobiCom'23]** Federated Few-Shot Learning for Mobile NLP ([paper](https://arxiv.org/abs/2212.05974), [code](https://github.com/UbiquitousLearning/FeS))
- FwdLLM: Efficient FedLLM using Forward Gradient ([paper](https://arxiv.org/abs/2308.13894), [code](https://github.com/UbiquitousLearning/FwdLLM))
- **[Electronics'24]** Forward Learning of Large Language Models by Consumer Devices ([paper](https://www.mdpi.com/2079-9292/13/2/402))
- Federated Fine-Tuning of LLMs on the Very Edge: The Good, the Bad, the Ugly ([paper](https://arxiv.org/pdf/2310.03150.pdf))
- Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes ([paper](https://arxiv.org/abs/2312.06353), [code](https://github.com/alibaba/FederatedScope/tree/FedKSeed))

## Mobile-Related Use-cases

This section includes paper that are mobile-related, but not necessarily run on device.

### Papers

#### 2025

- Small Language Models are the Future of Agentic AI ([paper](https://arxiv.org/abs/2506.02153))

#### 2024

- Ferret-UI: Grounded Mobile UI Understanding with Multimodal LLMs ([paper](https://arxiv.org/abs/2404.05719))
- Mobile-Agent: Autonomous Multi-Modal Mobile Device Agent with Visual Perception ([paper](https://arxiv.org/abs/2401.16158), [code](https://github.com/X-PLUG/MobileAgent))
- **[MobiCom'24]** MobileGPT: Augmenting LLM with Human-like App Memory for Mobile Task Automation ([paper](https://arxiv.org/abs/2312.03003))
- **[MobiCom'24]** AutoDroid: LLM-powered Task Automation in Android ([paper](https://arxiv.org/abs/2308.15272), [code](https://github.com/MobileLLM/AutoDroid))

#### 2023

- [NeurIPS'23] AndroidInTheWild: A Large-Scale Dataset For Android Device Control ([paper](https://arxiv.org/abs/2307.10088), [code](https://github.com/google-research/google-research/tree/master/android_in_the_wild))
- GPT-4V in Wonderland: Large Multimodal Models for Zero-Shot Smartphone GUI Navigation ([paper](https://arxiv.org/abs/2311.07562), [code](https://github.com/zzxslp/MM-Navigator))

#### Older

- [ACL'20] Mapping Natural Language Instructions to Mobile UI Action Sequences ([paper](https://arxiv.org/abs/2005.03776))

## Benchmarks

* [ExoLabs Benchmarks](https://benchmarks.exolabs.net/)
* [Qualcomm AI Hub](https://aihub.qualcomm.com/mobile/models)

## Leaderboards

* [HF Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
* [FlowerTune LLM Leaderboard](https://flower.ai/benchmarks/llm-leaderboard/)
* [MLPerf Inference: Mobile Benchmark Suite](https://mlcommons.org/benchmarks/inference-mobile/)

## Books and Courses

* [Edge AI Engineering](https://mjrovai.github.io/EdgeML_Made_Ease_ebook/) by Marcelo Rovai
* [Machine Learning Systems: Principles and Practices of Engineering Artificially Intelligent Systems](https://mlsysbook.ai/) by Vijay Janapa Reddi

## Industry Announcements

* [WWDC'24 - Apple Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
* [PyTorch Executorch Alpha](https://pytorch.org/blog/executorch-alpha/)
* [Google - LLMs On-Device with MediaPipe and TFLite](https://developers.googleblog.com/en/large-language-models-on-device-with-mediapipe-and-tensorflow-lite/)
* [Qualcomm - The future of AI is Hybrid](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Whitepaper-The-future-of-AI-is-hybrid-Part-1-Unlocking-the-generative-AI-future-with-on-device-and-hybrid-AI.pdf)
* [ARM - Generative AI on mobile](https://community.arm.com/arm-community-blogs/b/ai-and-ml-blog/posts/generative-ai-on-mobile-on-arm-cpu)

## Related Organized Workshops

* [TTODLer-FM @ ICML'25](https://ttodlerfm.gitlab.io/): Tiny Titans: The next wave of On-Device Learning for Foundational Models (TTODLer-FM)
* [ES-FoMO @ ICML'25](https://es-fomo.com/): Efficient Systems for Foundation Models
* [Binary Networks @ ICCV'25](https://binarynetworks.io/): Binary and Extreme Quantization for Computer Vision
* [SLLM @ ICLR'25](https://sites.google.com/view/sllm-iclr-2025): Workshop on Sparsity in LLMs: Deep Dive into Mixture of Experts, Quantization, Hardware, and Inference
* [MCDC @ ICLR'25](https://sites.google.com/view/mcdc2025/): Workshop on Modularity for Collaborative, Decentralized, and Continual Deep Learning
* [Adaptive Foundation Models @ NeurIPS'24](https://adaptive-foundation-models.org/)

## Related Awesome Repositories

If you want to read more about related topics, here are some tangential awesome repositories to visit:

* [NexaAI/Awesome-LLMs-on-device](https://github.com/NexaAI/Awesome-LLMs-on-device) on LLMs on Device
* [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) on Large Language Models
* [KennethanCeyer/awesome-llm](https://github.com/KennethanCeyer/awesome-llm) on  Large Language Models
* [HuangOwen/Awesome-LLM-Compression](https://github.com/HuangOwen/Awesome-LLM-Compression) on Large Language Model Compression
* [csarron/awesome-emdl](https://github.com/csarron/awesome-emdl) on Embedded and Mobile Deep Learning


## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.
