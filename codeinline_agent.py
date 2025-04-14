import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Generator
from smolagents.local_python_executor import LocalPythonExecutor
from transformers import BitsAndBytesConfig
import gradio as gr
import os

# Set up device configuration for multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    device_map = "auto"  # Let transformers automatically distribute the model
else:
    print("Only one GPU available, using it")
    device_map = "cuda:0"

model_name = "Qwen/Qwen2.5-Coder-7B"
model_name = "Qwen/QwQ-32B"

def sample(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, input_prompt: str, max_output=9000) -> Generator[str, None, None]:
    # Convert input to tokens and send to appropriate device(s)
    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
    if isinstance(device_map, str) and device_map != "auto":
        input_ids = input_ids.to(device_map)
    
    print(input_prompt, end="")
    for i in range(max_output):
        input = model.prepare_inputs_for_generation(input_ids)
        next_token_scores = model(**input)
        
        # Get the last token's logits
        last_token_logits = next_token_scores.logits[:, -1, :]
        
        # Convert to probabilities and sample
        probs = torch.nn.functional.softmax(last_token_logits, dim=1)
        output_id = torch.multinomial(probs, num_samples=1)
        
        # Decode the token
        output_str = tokenizer.decode(output_id[0])
        
        # Update input_ids with the new token
        if isinstance(device_map, str) and device_map != "auto":
            output_id = output_id.to(input_ids.device)
        input_ids = torch.cat([input_ids, output_id], dim=1)
        
        append_to_input = yield output_str
        
        if append_to_input is not None:
            # Add user feedback to the prompt
            new_prompt = tokenizer.decode(input_ids[0])
            new_prompt += append_to_input
            input_ids = tokenizer(new_prompt, return_tensors="pt").input_ids
            if isinstance(device_map, str) and device_map != "auto":
                input_ids = input_ids.to(device_map)
        
        if output_id[0] == tokenizer.eos_token_id:
            break

class CodeInlineAgent:
    def __init__(self, gpu_ids=None):
        # Configure which GPUs to use if specified
        self.gpu_ids = gpu_ids if gpu_ids is not None else list(range(torch.cuda.device_count()))
        
        if len(self.gpu_ids) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))
            print(f"Using GPUs: {self.gpu_ids}")
        
        # Configure quantization for efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load model with device map for multi-GPU
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map=device_map,  # Automatically distribute across available GPUs
            trust_remote_code=True,
        )
        
        # Load tokenizer
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            model_max_length=512,
            padding_side='left',
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize Python executor
        self.pyexp = LocalPythonExecutor(additional_authorized_imports=["yfinance", "str"])
        self.pyexp.send_tools({})

    def run(self, user_query : str):
        system_prompt = """You are an elite stock market analyst and trader with decades of experience in financial markets. 
Don't write functions, just write the code directly. Don't use pandas. 
You have access to yfinance. 
All code blocks written between ```python and ``` will get executed by a python interpreter and the result will be given to you.
Only write python code, and always use ```python at the start of code blocks
Write code to answer the user's question in these code blocks.
If the code execution fails, please rewrite and improve the code block. 
Please think step by step. Please use the python interpreter in your <think> block to do your thinking, always write all code blocks between`
for example:
<think>
Let me consider how to get the latest stock price of Apple.
```python
import yfinance as yf
apple = yf.Ticker("AAPL")
print(apple.history(period="1d"))
```
Successfully executed. Output from code block: $100

That means that the latest stock price of Apple is $100.
</think>
The latest stock price of Apple is $100.

Always write all code blocks between ```python and ```, even in <think> blocks. Don't write code in the answer. 
        """
        
        # Format the prompt for the model
        # prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        prompt: str = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}],
            tokenize=False,
            add_generation_prompt=True
        ) 
        total_output : str = prompt
        is_in_code_block = False
        code_block = ""
        gen = sample(self.model, self.tokenizer, prompt)
        for output in gen:
            if output=="<|im_end|>":
                break
            total_output += output
            
            if total_output.strip().endswith("```python"):
                is_in_code_block = True
                print("in code block")
            elif total_output.strip().endswith("```") and is_in_code_block:
                print("out code block")
                # print(code_block)
                is_in_code_block = False
                try:
                    output, execution_logs, is_final_answer = self.pyexp(code_block)
                    observation = "Successfully executed. Output from code block:\n" + str(execution_logs) + "\n"
                    # print(observation)
                except Exception as e:
                    observation = "\n"
                    if hasattr(self.pyexp, "state") and "_print_outputs" in self.pyexp.state:
                        execution_logs = str(self.pyexp.state["_print_outputs"])
                        if len(execution_logs) > 0:
                            observation += execution_logs
                    observation += "Failed. Please try another strategy. " + str(e) + ""
                        
                # print(observation)
                gen.send(observation)
                yield observation
                code_block = ""
            elif is_in_code_block and output != '`' and output != '``' and output != '```':
                code_block += output
            yield output
  
