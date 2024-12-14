from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info


# Load the model.
model_path = "/mnt/public/algm/models/Qwen2-VL-2B-Instruct"  # Change this to the path of the model.

llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 10, "video": 10},
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
    ignore_eos=True,
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "../data/sample_image.jpg",
                "min_pixels": 224 * 224,
                "max_pixels": 1024 * 1024,
            },
            {"type": "text", "text": prompt},
        ],
    },
]
processor = AutoProcessor.from_pretrained(model_path)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs = process_vision_info(messages)

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = [
        {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    for _ in range(num_requests)
]


outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
