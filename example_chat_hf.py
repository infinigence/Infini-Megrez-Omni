# -*- encoding: utf-8 -*-
# File: example_chat_hf.py
# Description: None

import torch
from transformers import AutoModelForCausalLM

path = "/mnt/algorithm/user_dir/zhoudong/workspace/models/megrez-o"  # Change this to the path of the model.

model = (
    AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    .eval()
    .cuda()
)
prompt = "hi" * (128 - 1) 
# Chat with text and image
messages = [
    {
        "role": "user",
        "content": {
            "text": prompt,
            "image": "./data/sample_image.jpg",
        },
    },
]

# Chat with audio and image
# messages = [
#     {
#         "role": "user",
#         "content": {
#             "image": "./data/sample_image.jpg",
#             "audio": "./data/sample_audio.m4a",
#         },
#     },
# ]

MAX_NEW_TOKENS = 100
response = model.chat(
    messages,
    sampling=False,
    max_new_tokens=MAX_NEW_TOKENS,
)
print(response)
