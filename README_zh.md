<div align="center">

# Megrez-3B-Omni: 首个端侧全模态理解开源模型

<p align="center">
    <img src="assets/megrez_logo.png" width="400"/>
<p>
<p align="center">
    🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Omni">Huggingface</a>&nbsp&nbsp | &nbsp&nbsp🤖<a href="https://www.modelscope.cn/models/InfiniAI/Megrez-3B-Omni">Modelscope</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://huggingface.co/spaces/Infinigence/Megrez-3B-Omni">Demo</a>&nbsp&nbsp | &nbsp&nbsp📖 <a href="assets/wechat-official.jpg">WeChat Official</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="assets/wechat-group.jpg">WeChat Groups</a>&nbsp&nbsp
</p>

<strong>中文 | [English](./README.md)</strong>

</div>

## 模型简介
Megrez-3B-Omni是由无问芯穹（[Infinigence AI](https://cloud.infini-ai.com/platform/ai)）研发的**端侧全模态**理解模型，基于无问大语言模型Megrez-3B-Instruct扩展，同时具备图片、文本、音频三种模态数据的理解分析能力，在三个方面均取得最优精度
- 在图像理解方面，基于SigLip-400M构建图像Token，在OpenCompass榜单上（综合8个主流多模态评测基准）平均得分66.2，超越LLaVA-NeXT-Yi-34B等更大参数规模的模型。Megrez-3B-Omni也是在MME、MMMU、OCRBench等测试集上目前精度最高的图像理解模型之一，在场景理解、OCR等方面具有良好表现。
- 在语言理解方面，Megrez-3B-Omni并未牺牲模型的文本处理能力，综合能力较单模态版本（Megrez-3B-Instruct）精度变化小于2%，保持在C-EVAL、MMLU/MMLU Pro、AlignBench等多个测试集上的最优精度优势，依然取得超越上一代14B模型的能力表现
- 在语音理解方面，采用Qwen2-Audio/whisper-large-v3的Encoder作为语音输入，支持中英文语音输入及多轮对话，支持对输入图片的语音提问，根据语音指令直接响应文本，在多项基准任务上取得了领先的结果

## 评测结果
- 左图为Megrez-3B-Omni与其他开源模型在主流图片多模态任务上的性能比较
- 右图为Megrez-3B-Omni在OpenCompass测试集上表现，图片引用自： [InternVL 2.5 Blog Post](https://internvl.github.io/blog/2024-12-05-InternVL-2.5/)*
<div style="display: flex; justify-content: space-between;">
  <img src="assets/multitask.jpg" alt="Image 1" style="width: 45%;">
  <img src="assets/opencompass.jpg" alt="Image 2" style="width: 45%;">
</div>

详细精度见 [Megrez-3B-Omni-HF](https://huggingface.co/Infinigence/Megrez-3B-Omni)

### 推理速度
|                | image_tokens | prefill (tokens/s) | decode (tokens/s) |
|----------------|:------------:|:------------------:|:-----------------:|
| Megrez-3B-Omni |      448     |       6312.66      |       1294.9      |
| Qwen2-VL-2B    |     1378     |       7349.39      |       685.66      |
| MiniCPM-V-2_6  |      448     |       2167.09      |       452.51      |

实验设置：
- 测试环境为NVIDIA H100下VLLM下输入128个Text token和一张 720*1480的图片，输出128个token，num_seqs固定为8。
- Qwen2-VL-2B的在此实验下的decode速度小于Megrez-3B-Omni，虽然其具备更小的基座LLM，但是编码上述大小图片后的image_token相较Megrez-3B-Omni较多，影响实际推理速度。

## 模型演示
【GIF】

## 安装
使用如下命令安装依赖：

```shell
pip install -r requirements.txt
```

音频功能依赖ffmpeg进行音频处理，如果您使用 Debian 相关的系统，可以通过以下命令安装：

```shell
sudo apt-get install ffmpeg
```

对于其他的操作系统，请参考 [ffmpeg 官方文档](https://ffmpeg.org/download.html) 进行安装。


## 模型推理

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

完整的示例见：[example_chat_hf.py](example_chat_hf.py).

### 使用 vLLM 进行推理
我们提供了一个基于 vLLM 框架的推理参考实现。您可以在 [vllm_demo/megrezo.py](vllm_demo/megrezo.py) 中找到模型定义。

推理步骤如下：

1. 安装 vLLM

```shell
pip install vllm==0.6.3.post1 flash_attn==2.5.8 xformers==0.0.27.post2
```

**注意**：使用 vLLM 推理需要安装特定版本的依赖，其他版本可能存在接口不一致的风险。有任何问题欢迎[提issue](https://github.com/infinigence/Infini-Megrez-Omni/issues/new)。

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

### WeiUI 演示

<div align="center" style="display: flex; justify-content: space-between;">
  <img src="assets/gradio_demo.jpg" style="width: 80%;">
</div>

### 在线 Demo

欢迎试用在线 Demo: [🤗Megrez-3B-Omni](https://huggingface.co/spaces/Infinigence/Megrez-3B-Omni)。

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

## 微调模型

我们提供了一个基于 [DeepSpeed](https://github.com/microsoft/DeepSpeed) 和 [accelerate](https://github.com/huggingface/accelerate) 的[微调示例](./finetune/)。

### 数据准备

我们基于[ALLaVA-4V/allava_laion](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/tree/main/allava_laion)构造了一个示例数据集：

- **对话**：[data/train/records.jsonl](./data/train/records.jsonl)
- **图片**：[data/train/images](./data/train/images)
- **音频**：[data/train/audio](./data/train/audio)，是通过将对话中的文本使用TTS转换为语音得到的。

您也可以按照上述格式准备自己的数据集。

### 依赖安装

```shell
pip install deepspeed accelerate
```

### 全参微调

使用如下命令运行我们的微调示例，请注意将脚本中的模型路径替换成您下载的模型路径。

```shell
cd finetune

sh finetune.sh
```

您可以通过设置`tune_vision_encoder`、`tune_vision_proj`、`tune_llm`、`tune_audio_encoder`、`tune_audio_proj`来选择需要微调的模块。

### 注意事项

- 推荐使用至少2张拥有80G显存的GPU进行微调。
- 在显存不足的情况下：
  - 请尝试调整`model_max_length`和`per_device_train_batch_size`。
  - 请尝试关闭需要微调的模块以便减少显存占用。
  - 请尝试调整deepspeed的`zero_optimization`参数来优化显存使用。
- 使用时
  - 请将图片尽量在首轮输入以保证推理效果，语音和文本无此限制，可以自由切换
  - 语音识别（ASR）场景下，只需要将content['text']修改为“将语音转化为文字。”
  - OCR场景下开启采样可能会引入语言模型幻觉导致的文字变化，可考虑关闭采样进行推理（sampling=False），但关闭采样可能引入模型复读

## 开源协议及使用声明

- **协议**：本仓库中代码依照 [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) 协议开源。
- **幻觉**：大模型天然存在幻觉问题，用户使用过程中请勿完全相信模型生成的内容。
- **价值观及安全性**：本模型已尽全力确保训练过程中使用的数据的合规性，但由于数据的大体量及复杂性，仍有可能存在一些无法预见的问题。如果出现使用本开源模型而导致的任何问题，包括但不限于数据安全问题、公共舆论风险，或模型被误导、滥用、传播或不当利用所带来的任何风险和问题，我们将不承担任何责任。
