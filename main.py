
import modal
import os
app = modal.App("codeinline-agent")
GPU_USED="A100-80gb"
def install_dependencies():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/QwQ-32B"
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

def install_dependencies_awq():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/QwQ-32B-AWQ"
    _ = AutoTokenizer.from_pretrained(model_name)
    _ = AutoModelForCausalLM.from_pretrained(model_name)

image = modal.Image.debian_slim(python_version="3.12").pip_install("transformers", "torch", "smolagents", "yfinance", "gradio")\
.pip_install("bitsandbytes", "accelerate", "litellm")\
.pip_install("vllm==0.8.2", "unsloth")\
.pip_install("autoawq")\
.run_commands("pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6")\
.add_local_python_source("codeinline_agent")\
.add_local_python_source("vllm_pipeline_patch")
# .run_function(install_dependencies)\
# .run_function(install_dependencies_awq,gpu=GPU_USED)\



@app.function(image=image, timeout=6000, gpu=GPU_USED)
def run_codeinline_agent(user_query : str):
    from codeinline_agent import CodeInlineAgent
    os.environ["TOGETHER_API_KEY"] = "tgp_v1_hGgxxj1Jk7ShXa3iFYAHOg7EbAS51lxcDZjZvb1Fols"
    agent = CodeInlineAgent()
    for output in agent.run(user_query):
        print(output, end="")

@app.local_entrypoint()
def main():
    run_codeinline_agent.remote("Which is a better stock to buy, Apple or Tesla?")


