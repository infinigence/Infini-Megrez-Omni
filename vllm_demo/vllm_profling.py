# -*- encoding: utf-8 -*-
# File: example_infer_vllm.py
# Description: None

from PIL import Image
from vllm import LLM
from vllm import ModelRegistry
from vllm import SamplingParams

from megrezo import MegrezOModel

ModelRegistry.register_model("MegrezO", MegrezOModel)

# Load the model.
model_path = "/mnt/algorithm/user_dir/zhoudong/workspace/models/megrez-o"  # Change this to the path of the model.
llm = LLM(
    model_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.9,
    max_num_seqs=8,
)

num_requests = 100
input_len = 128
output_length = 128
# prepare data 
prompt = "hi" * (input_len - 1) 
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=output_length,
    repetition_penalty=1.2,
    stop=["<|turn_end|>", "<|eos|>"],
    ignore_eos=True,
)

img = Image.open("../data/sample_image.jpg")

conversation = [
    {
        "role": "user",
        "content": {
            "text": prompt,
            "image": img,
        },
    },
]

# Convert the conversation to vLLM acceptable format.
prompt = llm.get_tokenizer().apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True,
)
vllm_inputs = [
    {
        "prompt": prompt,
        "multi_modal_data": {
            "image": img,
        },
    }
    for _ in range(num_requests)
]

# Generate the outputs.
outputs = llm.generate(
    vllm_inputs,
    sampling_params,
)

# Print the outputs.
# for output in outputs:
#     print(output.outputs[0].text)
