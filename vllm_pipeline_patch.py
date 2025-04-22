
import time
# If using hf transfer, you'll want to set HF_HUB_ENABLE_HF_TRANSFER env variable to 1.
# If running via terminal, you can just do `export HF_HUB_ENABLE_HF_TRANSFER=1`.
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from vllm import LLM, SamplingParams
import torch
import typing as t



def vllm_generate_iterator(
    self, prompt: str, /, *, echo: bool = False, stop: str  = None, stop_token_ids: t.List[int] = None, sampling_params=None, **attrs: t.Any
) -> t.Iterator[t.Dict[str, t.Any]]:
    request_id: str = attrs.pop('request_id', None)
    if request_id is None: raise ValueError('request_id must not be None.')
    if stop_token_ids is None: stop_token_ids = []
    stop_token_ids.append(self.tokenizer.eos_token_id)
    stop_ = set()
    if isinstance(stop, str) and stop != '': stop_.add(stop)
    elif isinstance(stop, list) and stop != []: stop_.update(stop)
    for tid in stop_token_ids:
        if tid: stop_.add(self.tokenizer.decode(tid))


    # if self.config['temperature'] <= 1e-5: top_p = 1.0
    # else: top_p = self.config['top_p']
    # config = self.config.model_construct_env(stop=list(stop_), top_p=top_p, **attrs)
    self.add_request(request_id=request_id, prompt=prompt, sampling_params=sampling_params)

    token_cache = []
    print_len = 0

    while self.has_unfinished_requests():
        for request_output in self.step():
            # Add the new tokens to the cache
            for output in request_output.outputs:
                text = output.text
                yield {'text': text, 'error_code': 0, 'num_tokens': len(output.token_ids)}

            if request_output.finished: break


class Pipeline:
    def __init__(self, **kwargs):
        self.llm = LLM(**kwargs)

    def __call__(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 50,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        prompt_template: str = "{prompt}",
        **kwargs,  # Passed to SamplingParams
    ):
        prompts = [
            (
                prompt_template.format(prompt=prompt),
                SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    **kwargs,
                )
            )
        ]
        start = time.time()
        while True:
            if prompts:
                prompt, sampling_params = prompts.pop(0)
                gen = vllm_generate_iterator(self.llm.llm_engine, prompt, echo=False, stop=None, stop_token_ids=None, sampling_params=sampling_params, request_id=0)
                last = ""
                for _, x in enumerate(gen):
                    if x['text'] == "":
                        continue
                    yield x['text'][len(last):]
                    last = x["text"]
                    num_tokens = x["num_tokens"]
                print(f"\nGenerated {num_tokens} tokens in {time.time() - start} seconds.")

                if not (self.llm.llm_engine.has_unfinished_requests() or prompts):
                    break