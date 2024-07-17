import tempfile

import torch
from diffusers import DiffusionPipeline
import dataclasses
from typing import Optional, List


@dataclasses.dataclass
class RiffusionPipelineOutput:
    audios: List
    duration_seconds: float


class RiffusionPipeline:
    def __init__(self, checkpoint: str = "riffusion/riffusion-model-v1", device: Optional[torch.device] = None):
        self.checkpoint = checkpoint
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        self.pipeline = DiffusionPipeline.from_pretrained(checkpoint, torch_dtype=dtype).to(self.device)

    @torch.no_grad()
    def generate_audio_from_prompt(self, prompt: str, duration: int, num_inference_steps: int = 50,
                                   guidance_scale: float = 7.5) -> RiffusionPipelineOutput:
        generator = torch.manual_seed(42) if torch.cuda.is_available() else None
        output = self.pipeline(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                               generator=generator)

        audio_clip = self.create_audio_clip(prompt, duration)

        return RiffusionPipelineOutput(audios=[audio_clip], duration_seconds=duration)

    def create_audio_clip(self, prompt, duration):
        # Заглушка для створення аудіо на основі текстового промпта
        import numpy as np

        # Генеруємо випадковий аудіосигнал (замість реальної генерації аудіо)
        audio_data = np.random.randn(int(duration * 44100))

        audio_path = f"{tempfile.gettempdir()}/generated_audio.wav"
        self.save_audio_clip(audio_data, audio_path, duration)

        return audio_path

    def save_audio_clip(self, audio_data, path, duration):
        import numpy as np
        import soundfile as sf

        audio_data = np.pad(audio_data, (0, int(duration * 44100) - len(audio_data)), mode='constant')

        sf.write(path, audio_data, 44100)
