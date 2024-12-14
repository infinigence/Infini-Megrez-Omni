from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/mnt/public/algm/models/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)


# default processer
processor = AutoProcessor.from_pretrained("/mnt/public/algm/models/Qwen2-VL-2B-Instruct")


prompt = "hi" * (128 - 1) 
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "../data/sample_image.jpg",
            },
            {"type": "text", "text": prompt},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
import pdb;pdb.set_trace()
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")
import pdb;pdb.set_trace()
# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)