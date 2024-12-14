from transformers import AutoTokenizer
from PIL import Image
from vllm import LLM, SamplingParams


model_path = "/mnt/public/algm/models/MiniCPM-V-2_6/"
image = Image.open("../data/sample_image.jpg").convert("RGB")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.9,
    max_num_seqs=8,
    trust_remote_code=True,
    max_model_len=4096
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


messages = [{
    "role":
    "user",
    "content":
    # Number of images
    "(<image>./</image>)" + \
    prompt
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Single Inference
llm_inputs = [{
    "prompt": prompt,
    "multi_modal_data": {
        "image": image
    },
} for _ in range(num_requests)]





outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
