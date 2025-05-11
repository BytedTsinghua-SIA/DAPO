from openai import AsyncOpenAI
import pandas as pd
import time
import asyncio
import tqdm
from tqdm.contrib.concurrent import process_map
from math_dapo import compute_score
import argparse

# Note: Ray Serve doesn't support all OpenAI client arguments and may ignore some.
client = AsyncOpenAI(
    # Replace the URL if deploying your app remotely
    # (e.g., on Anyscale or KubeRay).
    base_url=f"http://127.0.0.1:8000/v1",
    api_key="NOT A REAL KEY",
    timeout=36000,
)

async def chat(messages, temperature, top_p, max_tokens, model):
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_completion_tokens=max_tokens,
    )
    return completion

async def eval(tasks):
    start = time.time()
    with tqdm.tqdm(total=len(tasks)) as pbar:
        async def _tsk(coro):
            ret = await coro
            pbar.update(1)
            return ret
        tasks = [_tsk(t) for t in tasks]
        responses = await asyncio.gather(*tasks)
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return responses

def verify(arg):
    rsp, reward_model = arg
    return compute_score(rsp, reward_model['ground_truth'])

def main(args):
    df = pd.read_parquet(args.test_file)
    tasks = [chat(msg, args.temperature, args.top_p, args.max_tokens, args.model) for msg in df['prompt']]
    ret = asyncio.run(eval(tasks))
    df['output'] = [r.choices[0].message.content for r in ret]
    df['res'] = process_map(verify, df[['output', 'reward_model']].values, max_workers=50, chunksize=1)
    timestamp = time.strftime("%m%d_%H%M", time.localtime())
    df.to_parquet(f'eval_aime24_{timestamp}.parquet')

    score = 0
    for i, row in df.iterrows():
        score += row['res']['acc']
    avg_score = score / len(df)
    print(f"acc/mean@32: {avg_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=20480)
    parser.add_argument('--model', type=str, default='BytedTsinghua-SIA/DAPO-Qwen-32B')
    parser.add_argument('--test_file', type=str, default='aime-2024.parquet')
    args = parser.parse_args()
    print(args)
    main(args)
