# The AI Guide book

This repo is my note to the ever evolving Generative AI landscape.

## Terminologies
- Language model
- Large Language Models
- Pretrianed models
- Fine tuning
- Supervised fine tuning
- Instruction tuning
- Alignment
- Transfer Learning
- Agents
- Prompting
- Reinforcement Learning from Human Feedback
- Embeddings
- Diffusion models
- Latent Diffusion Models
- Instruction tuned models
- Chat tuned models

## Model Architecture
- Transformer
  - Encoder-Decoder
  - Decoder only
  - Encoder only
- Mixture of Experts
- Mamba
- State Space Models(SSM)
  
## Techniques

### Prompting Techniques
- Zero shot prompting
- Few shot prompting
- Chain of Thought
- ReACT
- Tree of Thought
- Multi Hop Question and Answering

### Fine tuning Techniques
- SFT
- Self instruct

### Parameter Effecient Fine Tuning Techniques
- LoRA
- QLoRA
- Adapters
- Prompt Tuning
- Prefix Tuning 

### Alignment techniques
- RLHF
- RLAIF
- [Proximal Policy Optimization(PPO)](https://en.wikipedia.org/wiki/Proximal_Policy_Optimization)
- Direct Preference Optimization (DPO)

### Quantization techniques
- GPTQ
- GGUF
- AWQ

### Metrics
- [ROGUE](https://aclanthology.org/W04-1013/)
- [BLEU](https://aclanthology.org/P02-1040.pdf)
- [GLUE](https://gluebenchmark.com/)

## Models

### Base models

- LLaMA1
- LLaMA2
- Mistral 7B
- Mixtral 8x7B
- GPT models
  - GPT
  - GPT2
  - GPT3
- MPT
- Gemini
- BERT

### Fine Tuned models
- [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
- [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)
- OpenHermes

- GPT-J
- [Guanaco](https://guanaco-model.github.io/)

### Diffusion models
- Stable diffusion
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- Denoising Diffusion Probabilistic Model

### Multi-modal models
- [LLaVA](https://llava-vl.github.io/)

### Music Gen models
- MusicLM

### Transformer Vision Models
- ViT
- Swim

### Vision Langauge Pretrained Models
- BLIP
- CLIP
- ALIGN
- ViLBERT

### Speech Recognition Models
- Whisper
- Conformer

### Speech Synthesis Models
- Voice Box
- Natural Speech

## Task
### Language
- QA
- Summurization

### Vision
- VQA
- VCR

## Prompt formats
Different LLMs have their formats for provide input to their fine tuned models. Read more [here](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [ChatML](https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/ai-services/openai/includes/chat-markup-language.md#chatml)
- LLaMA2 format
- Mistral instruct format



## Papers
[comment]: <> (A nice have column would be the corresponding papers with code column and a link or links to the paper explanation.)


|  name  |  Description   |
| :----: | :-------------:
| [Attention is all you need](https://arxiv.org/abs/1706.03762)     | The paper that introduced the transformer architecture.  |
|  [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | The paper that introduced intructgpt which was ground work for ChatGPT. |
|[Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)| |
|[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)| |
|[Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)| |
|[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)| The paper that introduced the **RAG** technique|
|[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | This paper instroduces the **Denoising Diffusion Probabilistic Model** which is used for image synthesis. |
|[Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)| This paper introduced the [CLIP](https://openai.com/research/clip) model. |
|[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://arxiv.org/abs/2112.10741)| This paper introduced the glide model. |
|[Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) | This paper introduced the concept of finetuning Language models. |
|[A Brief History of Prompt: Leveraging Language Models. (Through Advanced Prompting)](https://arxiv.org/abs/2310.04438)| |
|[A Comprehensive Overview of Large Language Models](https://arxiv.org/abs/2307.06435)||
|[First Tragedy, then Parse: History Repeats Itself in the New Era of Large Language Models](https://arxiv.org/abs/2311.05020)| This paper explains the existential crisis that most NLP currently face since the rise of LLMs. |
|[Large Language Model Alignment: A Survey](https://arxiv.org/abs/2309.15025)| A survey of LLM alignment. |
|[Mixtral of Experts](https://arxiv.org/abs/2401.04088)| Mistral paper on Mixtral 8x7B |
|[OUTRAGEOUSLY LARGE NEURAL NETWORKS:THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER](https://arxiv.org/abs/1701.06538)| This paper introduced the sparsely gated mixture of experts model. |
|[A General Survey on Attention Mechanisms in Deep Learning](https://arxiv.org/abs/2203.14263) | This paper is survey of the attention mechanism across domains in Deep Learning|
|[Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) | |
|[Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) | |
|[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | |
|[OpenAssistant Conversations -- Democratizing Large Language Model Alignment](https://arxiv.org/abs/2304.07327) | |
|[The Rise and Potential of Large Language Model Based Agents: A Survey](https://arxiv.org/pdf/2309.07864.pdf) | |


## Technologies

### Inference Runtimes
- Ollama
- llamacpp
- GPT4ALL

### Libaries
- Langchain
- LlamaIndex
- Huggingface
- Instructor
- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch): Implementation of **Denoising Diffusion Probabilistic Model**  in Pytorch.

### Multi Agent Frameworks
- [autogen](https://microsoft.github.io/autogen/)
- [CrewAI](https://docs.crewai.com/)

### Vector databases
- Weaviate
- Pinecone
- ChromaDB

## Blogs
- [DDPM in keras](https://keras.io/examples/generative/ddpm/): This tutorial implements the **Denoising Diffusion Probabilistic Model** in keras.
- [Introduction to Diffusion Models for Machine Learning](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/): Awesome blog bost introducing diffusion models.
- [instruction-tuning-vol-1](https://newsletter.ruder.io/p/instruction-tuning-vol-1): A walkththrough of various instruction tuning methods and datasets.
- [Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/): An illustrative guide to transformers.
- [Illustrated Stable Diffusion](http://jalammar.github.io/illustrated-stable-diffusion/): An illustrative guide to stable diffusion.
- [Painting with words a history of text to image ai](https://replicate.com/blog/painting-with-words-a-history-of-text-to-image-ai): Blog article details the history of text to image models.
- [Illustrating Reinforcement Learning from Human Feedback (RLHF) ](https://huggingface.co/blog/rlhf): Huggingface article on RLHF.
- [An Introduction to Diffusion Models and Stable Diffusion](https://blog.marvik.ai/2023/11/28/an-introduction-to-diffusion-models-and-stable-diffusion/): Anotehr awesome article on Diffusion models.
- [Mixture of Experts ml model guide](https://deepgram.com/learn/mixture-of-experts-ml-model-guide) by Deepgram.
- [Text to Video](https://huggingface.co/blog/text-to-video) explanation article by Huggingface.
- [A Visual Guide to MAMBA and State Space Models](https://maartengrootendorst.substack.com/p/a-visual-guide-to-mamba-and-state)
- [Which Quantization Method Is Right](https://maartengrootendorst.substack.com/p/which-quantization-method-is-right)
- [Why Are Sines and Cosines Used For Positional Encoding?](https://mfaizan.github.io/2023/04/02/sines.html)
- [Transformer Architecture: The Positional Encoding](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
- [An introduction to poisson flow generative models](https://www.assemblyai.com/blog/an-introduction-to-poisson-flow-generative-models/): An article introducing PFGM.

## Datasets
- [GLUE, the General Language Understanding Evaluation benchmark](https://huggingface.co/datasets/glue).
- [Wikitext](https://huggingface.co/datasets/wikitext): Contains text extracted from wikipedia.
- [IMDB](https://huggingface.co/datasets/imdb): This dataset is suitable for text binary classification.
- [Yelp review](https://huggingface.co/datasets/yelp_review_full): This dataset is suitable for text multi classification.
- [Text REtrieval Conference (TREC)](https://huggingface.co/datasets/trec): This dataset is for question classification.
- [AG news](https://huggingface.co/datasets/ag_news): This dataset is suitable for topic classification dataset. It contains 1 million news and their corresponding topic as labels. The labels fall into 5 classes.
-  [DPpedia 14](https://huggingface.co/datasets/fancyzhx/dbpedia_14): This dataset contains a subset of DBpedia dataset.


## Companies

|  company  |    Description |
| :----:    | :-------------:|
| [weaviate](https://weaviate.io/) | Vector database company |
| [Replicate](https://replicate.com/) | Deploys open source models |
| [ChromaDB](https://www.trychroma.com/) | Vector database company |
| [mistral](https://mistral.ai/)| Creators of the Mistral 7B and Mixtral 8x7B models|
| [deepgram](https://deepgram.com/)|A text to speech AI company |

## YouTube Channels

## Podcasts


## Other Resources/Inpirations
The following sites served as the inspiration for this repo.
- [FullStack Python](https://www.fullstackpython.com/): An all in one guide to fullstack development in python.
- [learnprompting](https://learnprompting.org/docs/intro)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [lilianweng blog on prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)

