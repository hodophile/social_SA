# Examples — one test per modality

Each script runs the full pipeline (preprocess → Qwen2.5-VLM-3B → structured JSON →
harm scoring → downstream action) against a sample asset in `assets/`.
All tests use the **mock Qwen2.5-VLM-3B**, so no GPU or checkpoint is required.

| Script           | Modality | Asset                  | What it checks                                   |
|------------------|----------|------------------------|--------------------------------------------------|
| `test_video.py`  | video    | `sample_video.mp4`     | frame sampling (1 fps) + audio extraction        |
| `test_audio.py`  | audio    | `sample_audio.wav`     | ffmpeg extraction → mono 16 kHz WAV, audio-only post |
| `test_image.py`  | image    | `sample_image.png`     | image loaded as a single frame                   |
| `test_text.py`   | text     | `sample_text.txt`      | text normalization + harmful-text → REVIEW path  |

## Run

```bash
pip install opencv-python   # required for frame/image loading
./run_all.sh                # or: python3 test_video.py (etc.)
```

## Regenerate assets

```bash
ffmpeg -y -f lavfi -i testsrc=size=640x360:rate=15:duration=4 \
       -f lavfi -i sine=frequency=440:duration=4 \
       -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest assets/sample_video.mp4
ffmpeg -y -f lavfi -i color=c=steelblue:size=640x360 -frames:v 1 assets/sample_image.png
ffmpeg -y -f lavfi -i "sine=frequency=660:duration=3" -ac 1 -ar 16000 assets/sample_audio.wav
```

## Real model inference

The examples default to `QwenVLM(mock=True)`. To use a real checkpoint
(`QwenVLM._load_model` is implemented via HuggingFace transformers,
default `Qwen/Qwen2.5-VL-3B-Instruct`):

```bash
pip install torch transformers accelerate opencv-python
python3 ../social_media_analyzer.py --real-model \
    --model <hf-id-or-local-path> \
    --video assets/sample_video.mp4 --text "Check out this clip!"
```
