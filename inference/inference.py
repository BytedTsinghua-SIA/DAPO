import torch
from vllm import SamplingParams, LLM
import json


def main():
    model = "BytedTsinghua-SIA/DAPO-Qwen-32B"

    llm = LLM(
        model=model,
        dtype=torch.bfloat16,
        tensor_parallel_size=8,
        gpu_memory_utilization=0.95
    )

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=0.7,
        max_tokens=32768
    )

    for example in json.load(open("examples.json", "r")):
        prompt = example["prompt"]
        answer = example["answer"]
        output = llm.generate(prompt, sampling_params)
        print(f"***QUESTION***:\n{prompt}\n***GROUND TRUTH***:\n{answer}\n***MODEL OUTPUT***:\n{output[0].outputs[0].text}\n")
        print("-"*100)

if __name__ == "__main__":
    main()