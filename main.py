from email.mime import audio
from io import BytesIO
import os
import gc

import numpy as np
from omnivoice import OmniVoice
from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
import soundfile as sf
import torch
from elevenlabs import ElevenLabs
from dotenv import load_dotenv

# Set multiprocessing start method to avoid semaphore leaks
if torch.multiprocessing.get_start_method(allow_none=True) is None:
    torch.multiprocessing.set_start_method('spawn', force=True)

load_dotenv()


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_best_device():
    """Auto-detect the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Language list — all 600+ supported languages
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)


def transcribe_elevenlabs(audio_path: str) -> str:
        elevenlabs_client = ElevenLabs(
                api_key=os.getenv("ELEVENLABS_API_KEY")
            )
        """Transcribe using ElevenLabs STT API"""
        if not elevenlabs_client:
            raise Exception("ElevenLabs client not initialized")

        # Check internet connection by attempting a quick request
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_bytes = audio_file.read()

            audio_data = BytesIO(audio_bytes)
            transcription = elevenlabs_client.speech_to_text.convert(
                file=audio_data,
                model_id="scribe_v2",
                language_code="eng",
            )
            return transcription.text
        except Exception as e:
            raise Exception(f"ElevenLabs API error: {e}")

def main(text, ref_audio, elevenlabs_transcribe=True):
    # if elevenlabs_transcribe:
    #     ref_text = transcribe_elevenlabs(ref_audio)
    #     print(f"Transcribed reference text: {ref_text}")
    # else:
    #     ref_text = None
    
    
    model = OmniVoice.from_pretrained(
        "./model",
        local_files_only=True,
        device_map="mps",
        dtype=torch.bfloat16,
        load_asr=False,
    )

    audio = model.generate(
        text,
        ref_audio=ref_audio,
        language="fr",
        ref_text="So this is basically a sample audio. We want to try out Pocket TT's voice cloning capabilities. We want to see how well it performs when it comes to cloning voices that are not American",  # optional
    )

    sf.write("clone_out.wav", audio[0], 24000)
    
    # Clean up resources to prevent semaphore leaks
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    
if __name__ == "__main__":
    try:
        main(
            "Salut ! Comment tu vas ? J'espère que tu passes une super journée ! Tu pourrais aussi égayer la mienne en appuyant sur le bouton s'abonner.",
            "austin.wav"
        )
    finally:
        # Final cleanup to ensure all resources are released
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()