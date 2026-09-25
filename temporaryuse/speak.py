import asyncio
import os
from pathlib import Path

import edge_tts

TELUGU_VOICE = "te-IN-MohanNeural"

_SPEAK_DIR = Path(__file__).resolve().parent
_DEFAULT_RELATIVE_AUDIO = _SPEAK_DIR / "static" / "audio.mp3"


def get_audio_output_path() -> Path:
    """
    Resolved path for the generated MP3.
    Override with env TEMPORARYUSE_AUDIO_PATH (absolute or relative cwd is expanded).
    Default: temporaryuse/static/audio.mp3 next to this file.
    """
    raw = os.getenv("TEMPORARYUSE_AUDIO_PATH", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_RELATIVE_AUDIO.resolve()


async def save_tts_mp3(
    text: str,
    output_path: str | Path | None = None,
    *,
    voice: str = TELUGU_VOICE,
    rate: str = "+20%",
    pitch: str = "-12Hz",
    volume: str = "+10%",
) -> Path:
    """Synthesize Telugu (or chosen voice) with Edge TTS and write MP3 to disk."""
    path = Path(output_path) if output_path else get_audio_output_path()
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
        volume=volume,
    )
    await communicate.save(str(path))
    return path


def save_tts_mp3_sync(
    text: str,
    output_path: str | Path | None = None,
    **kwargs,
) -> Path:
    """Sync wrapper for `save_tts_mp3` (e.g. TTS worker thread)."""
    return asyncio.run(save_tts_mp3(text, output_path, **kwargs))


if __name__ == "__main__":
    TEXT = """
నమస్కారం గణీ,
మీరు ఎలా ఉన్నారు?
"""
    out = asyncio.run(save_tts_mp3(TEXT))
    print("Saved:", out)