from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import hf_hub_download

MODEL_PATH = "facebook/Perception-LM-1B"
processor = AutoProcessor.from_pretrained(MODEL_PATH, use_fast=True)
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH).to("cuda")

video_file = hf_hub_download(
    repo_id="shumingh/perception_lm_test_videos",
    filename="GUWR5TyiY-M_000012_000022.mp4",
    repo_type="dataset",
)
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": video_file,
            },
            {"type": "text", "text": "Can you describe the video in detail?"},
        ],
    }
]
inputs = processor.apply_chat_template(
    [conversation],
    num_frames=32,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    video_load_backend="decord",
)
inputs = inputs.to(model.device)
generate_ids = model.generate(**inputs, max_new_tokens=256)
input_length = inputs["input_ids"].shape[1]
generate_ids_without_inputs = generate_ids[:, input_length:]

for output in processor.batch_decode(
    generate_ids_without_inputs, skip_special_tokens=True
):
    print(output)
