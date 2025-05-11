<div align='center'>
<h1>DAPO: an Open-source RL System from <br>ByteDance Seed and Tsinghua AIR</h1>

<!-- TODO:  Thread,Paper,Dataset,Weights-->
[![Paper](https://img.shields.io/badge/paper-5f16a8?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2503.14476)
[![Blog](https://img.shields.io/badge/Blog-3858bf?style=for-the-badge&logo=homepage&logoColor=white)](https://DAPO-SIA.github.io/)
[![Dataset](https://img.shields.io/badge/Datasets-4d8cd8?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
[![Weights](https://img.shields.io/badge/Model%20Weights-63cad3?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B)
<!-- [![Thread](https://img.shields.io/badge/Thread-91ded6?style=for-the-badge&logo=x&logoColor=white)](https://github.com/BytedTsinghua-SIA/DAPO) -->
</div>

> [!IMPORTANT]
> **🔥 News!!!**
> - [2025/05] We update the [wandb training record](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl?nw=wmb4qxfht0n) of full DAPO and the [checkpoint](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B) which achieved 50%+ on AIME 2024. We also provide [instructions](#evaluation-on-aime-2024) for evaluation on AIME 2024.
> - [2025/03] We release the training record of an early version of DAPO (w/o Token-level PG Loss & Dynamic Sampling), achieving 44% on AIME 2024, in [wandb](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl?nw=u7n2j5sht28).

We release a fully open-sourced system for large-scale LLM RL, including algorithm, code infrastructure, and dataset. The system achieves state-of-the-art large-scale LLM RL performance. We propose the **D**ecoupled Clip and **D**ynamic s**A**mpling **P**olicy **O**ptimization (**DAPO**) algorithm.
Through open-sourcing, we provide the broader research community and society with practical access to scalable reinforcement learning, enabling all to benefit from these advancements. Our system is based on the awesome [verl](https://github.com/volcengine/verl) framework. Thanks for their great work!

## Discussions Welcomed

🤗 If you have any questions about our paper, issues are welcomed and we could discuss there. Thank you!

## Key Results

### AIME 2024 Performance

🚀 **DAPO** achieves 50 points on AIME 2024 based on the Qwen2.5-32B base model, outperforming the previous SoTA DeepSeek-R1-Zero-Qwen-32B with 50% training steps.

![alt text](img/score.png)

### Metric Supervision during Training

1. **Length stability and growth**: The steady increase in response length allows for greater exploration, facilitating the model’s ability to learn more complex reasoning behaviors, ultimately contributing to training stability and performance improvement.

2. **Reward score stability**: A stable increase in the reward signal indicates that the model is successfully fitting the training distribution, ensuring that the learning process remains robust and consistent without significant fluctuations.

3. **Entropy and mean probability trend**: A controlled increase in entropy, after an initial decrease, ensures a healthy balance between exploration and exploitation, avoiding issues such as overfitting or excessive randomness, and promoting sustained model performance.

![alt text](img/dynamic.png)

## Model Use

We provide the model weights of [DAPO-Qwen-32B](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B), which is trained based on Qwen2.5-32B using the DAPO algorithm.

### Environment Setup

We recommend using conda to setup the environment:

```bash
conda create -n dapo python=3.10
conda activate dapo
pip3 install -r requirements.txt
```

### Inference

We provide the model inference code here:

```python
import torch
from transformers import AutoTokenizer
from vllm import SamplingParams, LLM

examples = [
    {
        "question": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nFind the largest possible real part of \\[(75+117i)z+\\frac{96+144i}{z}\\]where $z$ is a complex number with $|z|=4$.\n\nRemember to put your answer on its own line after \"Answer:\".",
        "answer": "540"
    },
    {
        "question": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nEvery morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.\n\nRemember to put your answer on its own line after \"Answer:\".",
        "answer": "204"
    },
    {
        "question": "Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\nLet $\\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\\mathcal{B}$. The value of $r^2$ can be written as $\\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.\n\nRemember to put your answer on its own line after \"Answer:\".",
        "answer": "721"
    }
]


def main():
    model = "BytedTsinghua-SIA/DAPO-Qwen-32B"

    tokenzier = AutoTokenizer.from_pretrained(model)

    llm = LLM(
        model=model,
        dtype=torch.bfloat16,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.95
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=20480
    )

    for example in examples:
        question = example["question"]
        answer = example["answer"]
        output = llm.generate(
                    prompts=tokenzier.apply_chat_template(conversation=[{"content": question, "role": "user"}],
                                                          add_generation_prompt=True,
                                                          tokenize=False),
                    sampling_params=sampling_params
                )
        print(f"***QUESTION***:\n{question}\n***GROUND TRUTH***:\n{answer}\n***MODEL OUTPUT***:\n{output[0].outputs[0].text}\n")
        print("-"*100)

if __name__ == "__main__":
    main()
```

### Evaluation on AIME 2024

To evaluate the model on AIME 2024, we deploy DAPO-Qwen-32B with Ray Serve and vLLM.

To load the model from Huggingface:

```bash
serve run eval.llm:build_app model=BytedTsinghua-SIA/DAPO-Qwen-32B tensor-parallel-size=8

# open another terminal
python eval/eval_aime24.py --temperature 1.0 --top_p 0.7 --max_tokens 20480 --model BytedTsinghua-SIA/DAPO-Qwen-32B --test_file eval/aime-2024.parquet
```

To load the model from local path:

```bash
serve run eval.llm:build_app model=aaa/bbb/ccc tensor-parallel-size=8

# open another terminal
python eval/eval_aime24.py --temperature 1.0 --top_p 0.7 --max_tokens 20480 --model ccc --test_file eval/aime-2024.parquet
```

## Reproducibility

To benefit the broader research community, we fully open-source the recipe of our RL training, including algorithm details, dataset, and infrastructures.

### Datasets
We provide training and validation datasets for DAPO training.

Training: [DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), a carefully curated and processed math dataset.
Validation: [AIME 2024](https://huggingface.co/datasets/BytedTsinghua-SIA/AIME-2024).

### Training

We provide the [out-of-the-box](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo) script for DAPO training reproduction. Quickstart and core code are mentioned in [README](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/README.md). These are scripts for:

- [Datasets Preparation](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/prepare_dapo_data.sh)
- [DAPO w/o Token-level PG Loss & Dynamic Sampling -- AIME 44](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/run_dapo_early_qwen2.5_32b.sh)
- [DAPO Full -- AIME 50](https://github.com/volcengine/verl/blob/gm-tyx/puffin/main/recipe/dapo/run_dapo_qwen2.5_32b.sh)

Note:

- The `DAPO w/o Token-level PG Loss & Dynamic Sampling -- AIME 44` script has been verified on the current verl and achieves 44 points on AIME, whose training record can be accessed in [wandb](https://wandb.ai/verl-org/DAPO%20Reproduction%20on%20verl?nw=u7n2j5sht28).

- The final performance of DAPO (50 on AIME) is achieved using the full DAPO algorithm based on our internal codebase, which includes heavy engineering optimization code based on verl. The `DAPO Full` script provides the command to run the full DAPO algorithm. But we still have not verified it on verl.

## Acknowledgement

We thank the [verl](https://github.com/volcengine/verl) for providing the awesome open-source RL infrastructure.

Our open-sourced experiments were conducted on the Volcano Engine Machine Learning Platform. We will provide a full reproduction guideline later on the Volcano Engine platform to help users replicate our experiments.

<!-- ## Citation -->
