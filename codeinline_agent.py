import abc
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Generator
from smolagents.local_python_executor import LocalPythonExecutor
from smolagents.tools import Tool
from smolagents import LiteLLMModel
from transformers import BitsAndBytesConfig
import gradio as gr
import os
from litellm import completion 

# Set up device configuration for multi-GPU
model_name = "Qwen/Qwen2.5-Coder-7B"
model_name = "Qwen/QwQ-32B" 

class AbstractSample:
    @abc.abstractmethod
    def sample(self, messages: list[dict[str, str]], ) -> Generator[str, None, None]:
        pass

class SampleFromHFDirectly(AbstractSample):
    def __init__(self, model_name : str, max_output=9000):
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
        self.max_output = max_output

        self.device_map = "auto"
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.device_map = "auto"  # Let transformers automatically distribute the model
        else:
            print("Only one GPU available, using it")
            self.device_map = "cuda:0"

    def sample(self, messages: list[dict[str, str]], ) -> Generator[str, None, None]:
        input_prompt: str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        ) 
        input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids
        if isinstance(self.device_map, str) and self.device_map != "auto":
            input_ids = input_ids.to(self.device_map)
        
        print(input_prompt, end="")
        for i in range(self.max_output):
            input = self.model.prepare_inputs_for_generation(input_ids)
            next_token_scores = self.model(**input)
            
            # Get the last token's logits
            last_token_logits = next_token_scores.logits[:, -1, :]
            
            # Convert to probabilities and sample
            probs = torch.nn.functional.softmax(last_token_logits, dim=1)
            output_id = torch.multinomial(probs, num_samples=1)
            
            # Decode the token
            output_str = self.tokenizer.decode(output_id[0])
            
            # Update input_ids with the new token
            if isinstance(self.device_map, str) and self.device_map != "auto":
                output_id = output_id.to(input_ids.device)
            input_ids = torch.cat([input_ids, output_id], dim=1)
            
            append_to_input = yield output_str
            
            if append_to_input is not None:
                # Add user feedback to the prompt
                new_prompt = self.tokenizer.decode(input_ids[0])
                new_prompt += append_to_input
                input_ids = self.tokenizer(new_prompt, return_tensors="pt").input_ids
                if isinstance(self.device_map, str) and self.device_map != "auto":
                    input_ids = input_ids.to(self.device_map)
            
            if output_id[0] == self.tokenizer.eos_token_id:
                break

class SampleFromLitellm(AbstractSample):
    def __init__(self, model : str, max_output=9000):
        self.max_output = max_output
        self.model_name = model_name

    def sample(self, messages: list[dict[str, str]], ) -> Generator[str, None, None]:
        messages = messages
        response = completion(model=self.model_name, messages=messages,max_completion_tokens=9000, stream=True)
        assistant_message = ""
        for chunk in response:
            msg = chunk["choices"][0]["delta"]["content"]
            if msg is not None:
                assistant_message += msg
                input = yield msg
                if input is not None:
                    assistant_message += input
                    messages.append({"role": "assistant", "content": assistant_message})
                    yield from self.sample(messages)
                    break
class CodeInlineAgent:
    def __init__(self, tools: dict = {}):
        # Configure which GPUs to use if specified
        self.sampler = SampleFromHFDirectly(model_name, max_output=9000)
        # self.sampler = SampleFromLitellm("together_ai/deepseek-ai/DeepSeek-R1", max_output=9000)
        # Initialize Python executor
        self.pyexp = LocalPythonExecutor(additional_authorized_imports=["yfinance", ])
        self.pyexp.send_tools(tools)
        self.tool_desc = self._generate_tool_descriptions(tools)

    def _generate_tool_descriptions(self, tools: dict) -> str:
        """Generate tool descriptions from a dictionary of tools.
        
        Args:
            tools: Dictionary of tools where keys are tool names and values are tool objects
            
        Returns:
            str: Formatted string containing tool descriptions
        """
        descriptions = []
        for tool in tools.values():
            # Generate function signature
            args = []
            for arg_name, arg_info in tool.inputs.items():
                args.append(f"{arg_name}: {arg_info.type}")
            signature = f"def {tool.name}({', '.join(args)}) -> {tool.output_type}:"
            
            # Generate docstring
            docstring = [f'    """{tool.description}', '', '    Args:']
            for arg_name, arg_info in tool.inputs.items():
                docstring.append(f'        {arg_name}: {arg_info.description}')
            docstring.append('    """')
            
            # Combine into full description
            descriptions.append('\n'.join([signature] + docstring))
        
        return '\n\n'.join(descriptions)

    def run(self, user_query : str):
        system_prompt = """You are an expert assistant who can solve any task using code blocks. You will be given a task to solve as best you can.
Don't write functions, just write the code directly. Don't use pandas. 
You have access to yfinance. 
All code blocks written between ```python and ``` will get executed by a python interpreter and the result will be given to you.
Only write python code, and always use ```python at the start of code blocks
Write code to answer the user's question in these code blocks.
If the code execution fails, please rewrite and improve the code block. 
Please think step by step. Please use the python interpreter in your <think> block to do your thinking, always write all code blocks between`
for example:
User: Generate an image of the oldest person in this document.
<think>
```python
answer = document_qa(document=document, question="Who is the oldest person mentioned?")
print(answer)
```
Successfully executed. Output from code block: John Doe

That means that the oldest person in the document is John Doe.
</think>
The oldest person in the document is John Doe.

On top of performing computations in the Python code snippets that you create, you only have access to these tools, behaving like regular python functions:
```python
{tool_desc}
```

Always write all code blocks between ```python and ```, even in <think> blocks. Don't write code in the answer. 
Only write code blocks between <think> and </think> tags, not in the answer.
Write several code blocks between <think> and </think> tags if needed.
Use only variables that you have defined!
Don't give up! You're in charge of solving the task, not providing directions to solve it.
        """.format(tool_desc=self.tool_desc)
        
        # Format the prompt for the model
        # prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"
        
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_query}]
        total_output : str = ""
        is_in_code_block = False
        code_block = ""
        gen = self.sampler.sample(messages)
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
  
