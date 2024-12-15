<div align="center">

# Megrez-3B-Omni: 软硬协同释放无穹端侧智能

<p align="center">
    <img src="assets/megrez_logo.png" width="400"/>
<p>
<p align="center">
        🤗 <a href="https://huggingface.co/Infinigence/Megrez-3B-Omni">Huggingface</a>&nbsp&nbsp | &nbsp&nbsp🤖<a href="https://www.modelscope.cn/models/InfiniAI/Megrez-3B-Omni">Modelscope</a>&nbsp&nbsp | &nbsp&nbsp🖥️ <a href="https://huggingface.co/Infinigence/Megrez-3B-Omni">Demo</a>&nbsp&nbsp | &nbsp&nbsp📖 <a href="https://cloud.infini-ai.com/assets/png/wechat_official_account.1f7e61401727063822266.png">WeChat Official</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://cloud.infini-ai.com/assets/png/wechat_community.7dbbc0b51727063822266.png">WeChat Groups</a>&nbsp&nbsp
</p>

<strong>[中文](./README_zh.md) | English</strong>

</div>

## Introduction

**Megrez-3B-Omni** is an edge-side multimodal understanding model developed by **Infinigence AI** ([Infinigence AI](https://cloud.infini-ai.com/platform/ai)). It is an extension of the Megrez-3B-Instruct model and supports comprehensive understanding and analysis of image, text, and audio modalities. The model achieves state-of-the-art accuracy in all three domains:

- Image Understanding: By utilizing SigLip-400M for constructing image tokens, Megrez-3B-Omni outperforms models with more parameters such as LLaVA-NeXT-Yi-34B, in overall performance. It is one of the highest-accuracy image understanding models across multiple mainstream benchmarks, including MME, MMVet, OCRBench, and MMMU. It demonstrates excellent performance in tasks such as scene understanding and OCR.

- Language Understanding: Megrez-3B-Omni retains exceptional text understanding capabilities without significant trade-offs. Compared to its single-modal counterpart (Megrez-3B-Instruct), its overall accuracy variation is less than 2%, maintaining state-of-the-art performance on benchmarks like C-EVAL, MMLU (Pro), and AlignBench. It continues to outperform previous-generation models with 14B parameters.

- Speech Understanding: Equipped with the encoder head of Whisper-large-v3 (~600M parameters), the model supports both Chinese and English speech input, multi-turn conversations, and voice-based questions about input images. It can directly respond to voice commands with text and has achieved leading results across multiple benchmark tasks.

## Model Info

<table>
  <thead>
    <tr>
      <th></th>
      <th>Language Module</th>
      <th>Vision Module</th>
      <th>Audio Module</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Architecture</td>
      <td>Llama-2 with GQA</td>
      <td>siglip-so400m</td>
      <td>Whisper-large-v3
(encoder-only)</td>
    </tr>
    <tr>
      <td># Params (Backbone)</td>
      <td>2.29B</td>
      <td>0.42B</td>
      <td>0.64B</td>
    </tr>
    <tr>
      <td>Connector</td>
      <td>-</td>
      <td>Cross Attention</td>
      <td>Linear</td>
    </tr>
    <tr>
      <td># Params (Others)</td>
      <td>Emb: 0.31B<br>Softmax: 0.31B</td>
      <td>Connector: 0.036B</td>
      <td>Connector: 0.003B</td>
    </tr>
    <tr>
      <td># Params (Total)</td>
      <td colspan="3">4B</td>
    </tr>
    <tr>
      <td># Vocab Size</td>
      <td>122880</td>
      <td>64 tokens/slice</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Context length</td>
      <td colspan="3">4K tokens</td>
    </tr>
    <tr>
      <td>Supported languages</td>
      <td colspan="3">Chinese & English</td>
    </tr>
  </tbody>
</table>

### Image Understanding

![OpencompassBmk](assets/opencompass.jpg)

|         model         |       basemodel       |  release time  | TF #Params（B） | #Params（B) | OpenCompass (Online) |   MME   | MMMU val |   OCRBench | Math-Vista-Mini | RealWorldQA | MMVet | hallusionBench | MMB TEST(en) | MMB TEST(zh) | TextVQA val | AI2D_TEST | MMstar | DocVQA_TEST |
|:---------------------:|:---------------------:|:----------:|:-------------:|:------------:|:------------------:|:-------:|:--------:|:----------:|:---------------:|:-----------:|:-----:|:--------------:|:------------:|:------------:|:-----------:|:---------:|:------:|:-----------:|
|     Megrez-3B-Omni    |       Megrez-3B       | 2024.12.16 |      2.29     |     3.376    |        65.5        | 2315.41 |   51.89  |    82.8    |        62       |    71.89    |  54.5 |      50.12     |     80.8     |     82.3     |     80.3    |   82.05   |  60.46 |    91.62    |
|         GPT-4o        |           -           | 2024.08.06 |       -       |       -      |        71.5        |  2328.7 |   69.2   |     736    |       61.3      |     75.4    |  69.1 |       55       |       -      |       -      |      -      |    84.6   |    -   |     92.8    |
|      GPT-4o mini      |           -           | 2024.08.06 |       -       |       -      |        64.1        |  2003.4 |    60    |     785    |       52.4      |     67.1    |  66.9 |      46.1      |       -      |       -      |      -      |    77.8   |    -   |      -      |
|  Qwen2-VL-2B-Instruct |       Qwen2-1.5B      | 2024.08.28 |      1.31     |     2.21     |        57.2        |   1872  |   41.1   |     794    |        43       |     62.9    |  49.5 |      41.7      |     74.9     |     73.5     |     79.7    |    74.7   |   48   |     90.1    |
|     InternVL2.5-2B    | Internlm2.5-1.8B-chat | 2024.12.06 |      1.89     |     2.21     |        59.9        |  2138.2 |   43.6   |     804    |       51.3      |     60.1    |  60.8 |      42.6      |     74.7     |     71.9     |     74.3    |    74.9   |  53.7  |     88.7    |
|      BlueLM-V-3B      |           -           | 2024.11.29 |      2.7      |      3.1     |        66.1        |    -    |   45.1   |     829    |       60.8      |     66.7    |  61.8 |       48       |      83      |     80.5     |     78.4    |    85.3   |  62.3  |     87.8    |
|     InternVL2.5-4B    |  Qwen2.5-3B-Instruct  | 2024.12.06 |      3.09     |     3.71     |        65.1        |  2337.5 |   52.3   |     828    |       60.5      |     64.3    |  60.6 |      46.3      |     81.1     |     79.3     |     76.8    |    81.4   |  58.3  |     91.6    |
|     Baichuan-Omni     |       Unknown-7B      | 2024.10.11 |       7       |       7      |          -         |  2186.9 |   47.3   |     700    |       51.9      |     62.6    |  65.4 |      47.8      |     76.2     |     74.9     |     74.3    |     -     |    -   |      -      |
|     MiniCPM-V-2.6     |        Qwen2-7B       | 2024.08.06 |      6.5      |      8.1     |        65.2        |  2348.4 |   49.8   |     852    |       60.6      |     69.7    |   60  |      48.1      |     81.2     |      79      |     80.1    |    82.1   |  57.26 |     90.8    |
|  Qwen2-VL-7B-Instruct |        Qwen2-7B       | 2024.08.28 |      6.5      |     8.29     |         67         |  2326.8 |   54.1   |     845    |       58.2      |     70.1    |   62  |      50.6      |      83      |     80.5     |     84.3    |     83    |  60.7  |     94.5    |
|  MiniCPM-Llama3-V-2.5 |   Llama3-Instruct 8B  | 2024.05.20 |       8       |     8.54     |        58.8        |  2024.6 |   45.8   |     725    |       54.3      |     63.5    |  52.8 |      42.4      |     77.2     |     74.2     |     76.6    |    78.4   |    -   |     84.8    |
|          VITA         |      Mixtral 8x7B     | 2024.08.12 |      12.9     |     12.9     |          -         |   2097  |   47.3   |     678    |       44.9      |      59     |  41.6 |      39.7      |     74.7     |     71.4     |     71.8    |     -     |    -   |      -      |
|       GLM-4V-9B       |        GLM-4-9B       | 2024.06.04 |      8.2      |     13.9     |        59.1        |  2018.8 |   46.9   |     776    |       51.1      |      -      |   58  |      46.6      |     81.1     |     79.4     |      -      |    81.1   |  58.7  |      -      |
|   LLaVA-NeXT-Yi-34B   |         Yi-34B        | 2024.01.18 |       34      |      34      |         55         |  2006.5 |   48.8   |     574    |       40.4      |      66     |  50.7 |      34.8      |     81.1     |      79      |     69.3    |    78.9   |  51.6  |      -      |
| Qwen2-VL-72B-Instruct |       Qwen2-72B       | 2024.08.28 |     70.21     |     73.4     |        74.8        |  2482.7 |   64.5   |     877    |       70.5      |     77.8    |   74  |      58.1      |     86.5     |     86.6     |     85.5    |    88.1   |  68.3  |     96.5    |

### Text Understanding

|                       |          |             |                                       | Chat&Instruction |                 |        | Zh&En Tasks |            |       |          |  Code |       | Math |       |
|:---------------------:|:--------:|:-----------:|:-------------------------------------:|:---------:|:---------------:|:------:|:-------------:|:----------:|:-----:|:--------:|:---------:|:-----:|:--------:|:-----:|
|         models        | Instruction |   Release Time  | Transformer #Params （w/o emb&softmax） |  MT-Bench | AlignBench (ZH) | IFEval |  C-EVAL (ZH)  | CMMLU (ZH) | MMLU  | MMLU-Pro | HumanEval |  MBPP |   GSM8K  |  MATH |
| Megrez-3B-Omni        |     Y    |  2024.12.16 |                  2.3                  |    8.4    |       6.5       |  66.5  |     84.0      |    75.3    | 73.3  |   45.2   |   72.6    | 60.6  |   63.8   | 27.3  |
| Megrez-3B-Instruct    |     Y    |  2024.12.16 |                  2.3                  |   8.64    |      7.06       |  68.6  |     84.8      |    74.7    | 72.8  |   46.1   |   78.7    | 71.0  |   65.5   | 28.3  |
| Baichuan-Omni         |     Y    |  2024.10.11 |                  7.0                  |     -     |        -        |    -   |     68.9      |    72.2    |  65.3 |     -    |     -     |   -   |     -    |   -   |
| VITA                  |     Y    |  2024.08.12 |                 12.9                  |     -     |        -        |    -   |     56.7      |    46.6    | 71.0  |     -    |     -     |   -   |   75.7   |   -   |
| Qwen1.5-7B            |          |  2024.02.04 |                  6.5                  |     -     |        -        |    -   |     74.1      |    73.1    | 61.0  |   29.9   |   36.0    | 51.6  |   62.5   | 20.3  |
| Qwen1.5-7B-Chat       |     Y    |  2024.02.04 |                  6.5                  |   7.60    |      6.20       |    -   |     67.3      |      -     | 59.5  |   29.1   |   46.3    | 48.9  |   60.3   | 23.2  |
| Qwen1.5-14B           |          |  2024.02.04 |                 12.6                  |     -     |        -        |    -   |     78.7      |    77.6    | 67.6  |     -    |   37.8    | 44.0  |   70.1   | 29.2  |
| Qwen1.5-14B-Chat      |     Y    |  2024.02.04 |                 12.6                  |    7.9    |        -        |    -   |       -       |      -     |   -   |     -    |     -     |   -   |     -    |   -   |
| Qwen2-7B              |          |  2024.06.07 |                  6.5                  |     -     |        -        |    -   |     83.2      |    83.9    | 70.3  |   40.0   |   51.2    | 65.9  |   79.9   | 44.2  |
| Qwen2-7b-Instruct     |     Y    |  2024.06.07 |                  6.5                  |   8.41    |      7.21       |  51.4  |     80.9      |    77.2    | 70.5  |   44.1   |   79.9    | 67.2  |   85.7   | 52.9  |
| Qwen2.5-3B-Instruct   |     Y    |  2024.9.19  |                  2.8                  |     -     |        -        |    -   |       -       |      -     |   -   |   43.7   |   74.4    | 72.7  |   86.7   | 65.9  |
| Qwen2.5-7B            |          |  2024.9.19  |                  6.5                  |     -     |        -        |    -   |       -       |      -     | 74.2  |   45.0   |   57.9    | 74.9  |   85.4   | 49.8  |
| Qwen2.5-7B-Instruct   |     Y    |  2024.09.19 |                  6.5                  |   8.75    |        -        |  74.9  |       -       |      -     |   -   |   56.3   |   84.8    | 79.2  |   91.6   | 75.5  |
| Llama-3.1-8B          |          |  2024.07.23 |                  7.0                  |    8.3    |       5.7       |  71.5  |     55.2      |    55.8    | 66.7  |   37.1   |     -     |   -   |   84.5   | 51.9  |
| Llama-3.2-3B          |          |  2024.09.25 |                  2.8                  |     -     |        -        |  77.4  |       -       |      -     | 63.4  |     -    |     -     |   -   |   77.7   | 48.0  |
| Phi-3.5-mini-instruct |     Y    |  2024.08.23 |                  3.6                  |    8.6    |       5.7       |  49.4  |     46.1      |    46.9    | 69.0  |   47.4   |   62.8    | 69.6  |   86.2   | 48.5  |
| MiniCPM3-4B           |     Y    |  2024.09.05 |                  3.9                  |   8.41    |      6.74       |  68.4  |     73.6      |    73.3    | 67.2  |     -    |   74.4    | 72.5  |   81.1   | 46.6  |
| Yi-1.5-6B-Chat        |     Y    |  2024.05.11 |                  5.5                  |   7.50    |      6.20       |    -   |     74.2      |    74.7    | 61.0  |     -    |   64.0    | 70.9  |   78.9   | 40.5  |
| GLM-4-9B-chat         |     Y    |  2024.06.04 |                  8.2                  |   8.35    |      7.01       |  64.5  |     75.6      |    71.5    | 72.4  |     -    |   71.8    |   -   |   79.6   | 50.6  |
| Baichuan2-13B-Base    |          |  2023.09.06 |                 12.6                  |     -     |      5.25       |    -   |     58.1      |    62.0    | 59.2  |     -    |   17.1    | 30.2  |   52.8   | 10.1  |

- The metrics for the Qwen2-1.5B model differ between the original paper and the Qwen2.5 report. Currently, the accuracy figures from the original paper are being used.

### Audio Understanding

|       Model      |     Base model     | Realease Time | Fleurs test-zh | WenetSpeech test_net | WenetSpeech test_meeting |
|:----------------:|:------------------:|:-------------:|:--------------:|:--------------------:|:------------------------:|
| Whisper-large-v3 |          -         |   2023.11.06  |      12.4      |         17.5         |           30.8           |
|  Qwen2-Audio-7B  |      Qwen2-7B      |   2024.08.09  |        9       |          11          |           10.7           |
|  Baichuan2-omni  |     Unknown-7B     |   2024.10.11  |        7       |          6.9         |            8.4           |
|       VITA       |    Mixtral 8x7B    |   2024.08.12  |        -       |      -/12.2(CER)     |        -/16.5(CER)       |
|  Megrez-3B-Omni  | Megrez-3B-Instruct |   2024.12.16  |      10.8      |         45.08        |           16.44          |

### Inference Speed

|                | image_tokens | prefill (tokens/s) | decode (tokens/s) |
|----------------|:------------:|:------------------:|:-----------------:|
| Megrez-3B-Omni |      448     |       6312.66      |       1294.9      |
| Qwen2-VL-2B    |     1378     |       7349.39      |       685.66      |
| MiniCPM-V-2_6  |      448     |       2167.09      |       452.51      |

Setup:  

- The testing environment utilizes an NVIDIA H100 GPU with VLLM. Each test includes 128 text tokens and a 720×1480 image as input, producing 128 output tokens, with `num_seqs` fixed at 8.  
- Under this setup, the decoding speed of **Qwen2-VL-2B** is slower than **Megrez-3B-Omni**, despite having a smaller base LLM. This is due to the larger number of image tokens generated when encoding images of the specified size, which impacts actual inference speed.  

## Install

```shell
pip install -r requirements.txt
```

## Fine-Tuning the Model

We provide a [fine-tuning example](./finetune/) based on [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [accelerate](https://github.com/huggingface/accelerate).

### Data Preparation

We have constructed a sample dataset based on [ALLaVA-4V/allava_laion](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V/tree/main/allava_laion) dataset:  

- **Dialogue**: [data/train/records.jsonl](./data/train/records.jsonl)  
- **Images**: [data/train/images](./data/train/images)  
- **Audio**: [data/train/audio](./data/train/audio), created by converting dialogue text into speech using TTS.  

You can also prepare your own dataset following the same format.

### Dependencies Installation

Install the required dependencies with the following command:  

```bash
pip install deepspeed accelerate
```

### Full-Parameter Fine-Tuning

To run the fine-tuning example, execute the following commands. Be sure to replace the model path in the script with the path to your downloaded model.  

```bash
cd finetune

sh finetune.sh
```

You can customize the modules to fine-tune by setting the parameters:  
`tune_vision_encoder`, `tune_vision_proj`, `tune_llm`, `tune_audio_encoder`, and `tune_audio_proj`.

### Notes

1. **Recommended Hardware**: Please use at least two GPUs with 80GB memory for fine-tuning.  
2. **If GPU memory is insufficient**:  
   - Adjust the `model_max_length` and `per_device_train_batch_size` parameters.  
   - Disable specific modules for fine-tuning to reduce memory usage.  
   - Optimize memory consumption by configuring the `zero_optimization` parameters in DeepSpeed.  

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

## Open Source License and Usage Statement

- **License**: The code in this repository is open-sourced under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) license.  
- **Hallucination**: Large models inherently suffer from hallucination issues. Users are advised not to fully trust the content generated by the model. For more factual outputs, we recommend utilizing our WebSearch functionality, as detailed [here](xxxx).  
- **Mathematics & Reasoning**: Smaller models are more prone to errors in mathematical calculations and reasoning chains, leading to incorrect final results. Notably, the softmax distribution of smaller models is less sharp compared to larger models, making them more likely to produce inconsistent results under higher temperature settings, especially for deterministic tasks like mathematics or reasoning. We recommend lowering the temperature or performing multiple inference runs for verification in such cases.  
- **System Prompt**: Similar to most models, we recommend using the default system prompt in the `chat_template` configuration file for a stable and balanced experience. In this release, role-playing and domain-specific application capabilities have been de-emphasized. If users require specific domain applications, we suggest fine-tuning the model accordingly.  
- **Values and Safety**: While we have made every effort to ensure compliance of the data used during training, the large volume and complexity of the data may still lead to unforeseen issues. We disclaim any liability for problems arising from the use of this open-source model, including but not limited to data security issues, public opinion risks, or risks and problems caused by misleading, misuse, propagation, or improper utilization of the model.  

## Contact Us

![wechat](assets/wechat.jpg)
