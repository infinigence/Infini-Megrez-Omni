# Megrez-O

<strong>[中文](./README_zh.md) |
English</strong>

## Install

```shell
pip install -r requirements.txt
```

## Finetune

{{TBD}}

## Inference

### Conversation with Multimodal Data

You can use the following script to chat with our model. Note that you should replace `PATH_TO_PRETRAINED_MODEL` with the path to the downloaded model checkpoint.

```python
import torch
from transformers import AutoModelForCausalLM

path = "{{PATH_TO_PRETRAINED_MODEL}}"  # Change this to the path of the model.

model = (
    AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    .eval()
    .cuda()
)

messages = [
    {
        "role": "user",
        "content": {
            "text": "Please describe the content of the image.",
            "image": "./data/sample_image.jpg",
        },
    },
]

MAX_NEW_TOKENS = 100
response = model.chat(
    messages,
    sampling=False,
    max_new_tokens=MAX_NEW_TOKENS,
)
print(response)
```

You can also find a complete script in [example_chat_hf.py](example_chat_hf.py).

### Inference with vLLM

We provide a reference implementation of inference with vLLM framework. You can find the model definition in [vllm_demo/megrezo.py](vllm_demo/megrezo.py).

1. Install vLLM

```shell
pip install vllm==0.6.3.post1 flash_attn==2.5.8 xformers==0.0.27.post2
```

2. Run the inference script

Since vLLM does not officially support MegrezO yet, you need to import the module first:

```python
from vllm import ModelRegistry
from megrezo import MegrezOModel

ModelRegistry.register_model("MegrezO", MegrezOModel)
```

Then, you can run inference with the following code:

```python
from PIL import Image
from vllm import LLM
from vllm import SamplingParams


# Load the model.
model_path = "{{PATH_TO_HF_PRETRAINED_MODEL}}"  # Change this to the path of the model.
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
```

You can find a complete script in [vllm_demo/example_infer_vllm.py](vllm_demo/example_infer_vllm.py).

## Chat with MegrezO using Gradio

We provide online and local demos powered by Hugging Face Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>.

### Online Demo

Please try out our online Demo here: {{TBD}}

### Local WebUI Demo
  
You can easily deploy your own local WebUI to chat with MegrezO using Gradio.

1. Install dependencies:

```shell
pip install -r requirements.txt
```

2. Launch the Gradio app.

You need to specify the `model_path` and `port` in the command line. The `model_path` is the path to the model checkpoint, and the `port` is the port number for the local server. By default, the `port` is `7860`.

```shell
python gradio_app.py --model_path {model_path} --port {port}
```

Then, you can visit `http://localhost:7860` in your browser to interact with the model.

Feel free to modify the `gradio_app.py` to customize the input and output interfaces. For more information, please refer to the [Gradio documentation](https://gradio.app/docs).
