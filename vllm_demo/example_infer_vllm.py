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
# model_path = "{{PATH_TO_HF_PRETRAINED_MODEL}}"  # Change this to the path of the model.
model_path = "/mnt/algorithm/user_dir/zhoudong/workspace/models/megrez-o"  # Change this to the path of the model.
llm = LLM(
    model_path,
    trust_remote_code=True,
    gpu_memory_utilization=0.5,
)

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1000,
    repetition_penalty=1.2,
    stop=["<|turn_end|>", "<|eos|>"],
)

img = Image.open("../data/sample_image.jpg")

conversation = [
    {
        "role": "user",
        "content": {
            "text": "图片的内容是什么？",
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
]

# Generate the outputs.
outputs = llm.generate(
    vllm_inputs,
    sampling_params,
)

# Print the outputs.
for output in outputs:
    print(output.outputs[0].text)
