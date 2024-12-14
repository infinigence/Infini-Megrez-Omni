# Megrez-O

<strong>中文 |
[English](./README.md)</strong>

## 安装

使用如下命令安装依赖：

```shell
pip install -r requirements.txt
```

## 微调模型

{{TBD}}

## 推理

### 使用多模态数据进行多轮对话

请使用如下脚本进行推理。请将 `PATH_TO_PRETRAINED_MODEL` 替换为下载的模型权重的路径。

```python
import torch
from transformers import AutoModelForCausalLM

path = "{{PATH_TO_PRETRAINED_MODEL}}"  # 更改为模型的路径

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

完整的实例见：[example_chat_hf.py](example_chat_hf.py).

### 使用 vLLM 进行推理

我们提供了一个基于 vLLM 框架的推理参考实现。您可以在 [vllm_demo/megrezo.py](vllm_demo/megrezo.py) 中找到模型定义。

推理步骤如下：

1. 安装 vLLM

注意，我们需要安装特定版本的依赖：

```shell
pip install vllm==0.6.3.post1 flash_attn==2.5.8 xformers==0.0.27.post2
```

2. 运行推理脚本

vLLM 尚未正式支持 MegrezO，因此您需要先导入我们定义的模块：

```python
from vllm import ModelRegistry
from megrezo import MegrezOModel

ModelRegistry.register_model("MegrezO", MegrezOModel)
```

然后，您可以使用以下代码运行推理：

```python
from PIL import Image
from vllm import LLM
from vllm import SamplingParams


model_path = "{{PATH_TO_HF_PRETRAINED_MODEL}}"  # 更改为模型的路径
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

# 将对话转换为 vLLM 可接受的格式。
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

# 生成输出
outputs = llm.generate(
    vllm_inputs,
    sampling_params,
)

# 打印输出
for output in outputs:
    print(output.outputs[0].text)
```

完整的示例见：[vllm_demo/example_infer_vllm.py](vllm_demo/example_infer_vllm.py).

## 使用 Gradio 与 MegrezO 对话

我们提供基于 Hugging Face Gradio <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 实现的在线和本地 Demo。

### 在线 Demo

欢迎试用在线 Demo: {{TBD}}。

### 本地 Demo
  
使用如下命令部署本地 Gradio 应用：

1. 安装依赖:

```shell
pip install -r requirements.txt
```

2. 启动 Gradio 应用

您需要在命令行中指定 `model_path` 和 `port`。`model_path` 是模型的路径，`port` 是本地服务器的端口号。默认情况下，`port` 是 `7860`。

```shell
python gradio_app.py --model_path {model_path} --port {port}
```

然后，您可以在浏览器中访问 `http://localhost:7860` 与模型对话。

如需自定义输入和输出接口，请修改 `gradio_app.py`。更多信息请参考 [Gradio 文档](https://gradio.app/docs)。
