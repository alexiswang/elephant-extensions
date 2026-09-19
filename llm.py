import asyncio
import os

import httpx
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio

from utils import results_to_dataframe  

load_dotenv()

DEFAULT_RATE_LIMIT_PER_MINUTE = 10


@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=1, min=4, max=10),
       retry=retry_if_exception_type(httpx.HTTPStatusError),
       reraise=True)
async def call_openrouter_async(client, model, prompt, api_key=None, max_tokens=2048, temperature=1.0):
    """
    Async call to any model through OpenRouter API using tenacity decorator
    """
    if api_key is None:
        api_key = os.getenv('OPENROUTER_API_KEY')

    if not api_key:
        raise ValueError("API key must be provided or set in OPENROUTER_API_KEY environment variable")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = await client.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()

    return result


async def process_single_sample_async(client, model, prompt, temperature, sample_idx, input_idx, semaphore, rate_limiter):
    """Process a single sample - tenacity in call_openrouter_async handles retries"""
    async with rate_limiter:
        async with semaphore:
            try:
                results = await call_openrouter_async(
                    client=client,
                    model=model,
                    prompt=prompt,
                    temperature=temperature
                )

                message = results['choices'][0]['message']
                answer = message.get('content', '')
                return {
                    'input_idx': input_idx,
                    'sample_idx': sample_idx,
                    'answer': answer,
                    'success': True,
                    'error': None
                }
            except Exception as e:
                return {
                    'input_idx': input_idx,
                    'sample_idx': sample_idx,
                    'answer': None,
                    'success': False,
                    'error': str(e)
                }


async def batch_process_async(
    prompts,
    model,
    prompt_template=None,
    num_samples=1,
    temperature=1.0,
    max_concurrent=20,
    rate_limit_per_minute=DEFAULT_RATE_LIMIT_PER_MINUTE
):
    """
    Async batch processing with progress bar
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    rate_limiter = AsyncLimiter(rate_limit_per_minute, time_period=60)
    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = []
        for input_idx, prompt in enumerate(prompts):
            if prompt_template:
                formatted_prompt = prompt_template.format(input=prompt)
            else:
                formatted_prompt = prompt

            for sample_idx in range(num_samples):
                task = process_single_sample_async(
                    client, model, formatted_prompt, temperature, sample_idx, input_idx, semaphore, rate_limiter
                )
                tasks.append(task)

        results = await tqdm_asyncio.gather(*tasks, desc=f"Processing {len(prompts)} questions")

    final_results = {}
    for result in results:
        input_idx = result['input_idx']
        if input_idx not in final_results:
            final_results[input_idx] = {
                'input_idx': input_idx,
                'input': prompts[input_idx],
                'samples': []
            }
        final_results[input_idx]['samples'].append(result)

    return [final_results[idx] for idx in sorted(final_results.keys())]
