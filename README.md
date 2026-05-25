# OmniVoice Document Reader

A local document reader that synthesizes full-length documents in your own voice — or any built-in voice — sentence by sentence, with real-time highlighting and seamless browser playback.

Built on top of [OmniVoice](https://github.com/k2-fsa/OmniVoice) — the original model repo by k2-fsa.

---

## What it does

- Upload a PDF, DOCX, Markdown, or plain-text document
- Choose a TTS engine: **OmniVoice** (zero-shot voice cloning, 600+ languages) or **Supertonic 3** (9 built-in voices, 31 languages)
- Hit Play — audio streams sentence-pair by sentence-pair as it generates
- The browser player grows seamlessly as new audio is appended; playback never restarts from zero
- The document view highlights the current sentences and marks completed ones
- A progress bar tracks overall position through the document
- Pause/Resume/Stop all work mid-generation

---

## Requirements

```
pip install omnivoice fastapi uvicorn soundfile python-dotenv elevenlabs nltk
pip install pymupdf python-docx          # PDF and DOCX parsing
```

You also need:
- `./model/` — local OmniVoice model weights (downloaded from HuggingFace `k2-fsa/OmniVoice`)
- `./supertonic3/` — Supertonic 3 ONNX model directory (optional; Supertonic tab is hidden if missing)
- A `.env` file with `ELEVENLABS_API_KEY=...` if you want automatic reference audio transcription

---

## server.py — Main app

`server.py` is the primary application. It serves a full browser UI and streams generated audio over WebSocket.

```bash
python server.py
# then open http://localhost:7860
```

### How to use it

1. **Choose engine** — OmniVoice for voice cloning, Supertonic for built-in voices
2. **OmniVoice — Voice Clone mode**: upload a 3–10 second WAV of the voice you want to clone. The app calls ElevenLabs Scribe v2 to auto-transcribe it; you can also type the transcript manually.
3. **OmniVoice — Voice Design mode**: check "Voice Design Mode" and type a description like `female, young adult, american accent, moderate pitch`. No reference audio needed.
4. **Supertonic**: pick a speaker (M1–M4, F1–F5) and language from the dropdowns.
5. **Upload a document** (PDF, DOCX, .md, or .txt).
6. **Press Play** — audio starts generating and playing immediately.

Generated audio is saved incrementally under `outputs/<doc>_<voice>/`. Each session ends with a `session_final.wav` containing the full document audio.

### server.py architecture

| Section | What it does |
|---|---|
| **1 — Model loading** | Loads OmniVoice from `./model` at startup using MPS (Apple Silicon) with `bfloat16`. Attempts to load Supertonic 3 from `./supertonic3`; gracefully disables it if the directory is missing or import fails. |
| **2 — Document parsing** | `parse_pdf`, `parse_docx`, `parse_markdown` each walk their respective format and return a list of `DocBlock` objects tagged as `h1/h2/h3/p`. `assign_sentences` runs NLTK sentence tokenization over all paragraph blocks, assigning each sentence a flat global index. `build_doc_html` renders the blocks to HTML with `<span id="s-N">` tags on each sentence for live highlighting. |
| **3 — Audio generation** | `gen_omni` calls `omni_tts.generate()` with either a reference clip (voice clone) or an `instruct` string (voice design). `gen_supertonic` calls `st_tts.get_voice_style(voice)` then `st_tts.synthesize()`. `_normalise` converts any torch Tensor or int16 array to a float32 mono numpy array. `transcribe_elevenlabs` sends a WAV to ElevenLabs Scribe v2 and returns the transcript string. |
| **4 — Session state** | Module-level globals hold the current document sentences, parsed blocks, file paths, and two threading primitives: `_stop_flag` (Event) to abort generation, and `_playback_event` (Event) to pause/resume the producer thread. |
| **5 — FastAPI app** | A single `FastAPI()` instance. The `outputs/` directory is mounted as a static file server so the browser can fetch WAV files directly via `/outputs/...` URLs. |
| **6 — HTML page** | The entire UI is a single self-contained HTML string (`_HTML_TEMPLATE`) injected with the language list at startup. It contains: a sidebar with engine/voice/document controls; a document viewer with sentence spans; a progress bar; and a persistent `<audio id="player">` element. The JS `loadAudio(url, duration)` function handles every audio update — it captures the current playback position before swapping `src`, then restores it on the `canplay` event using `fastSeek`. The `userPaused` flag ensures that a natural audio-end triggers auto-resume on the next chunk, but a manual Pause click does not. |
| **7 — Routes** | `GET /` returns the HTML page. `POST /upload/doc` saves the file to a temp path, parses it, stores sentences globally, and returns rendered HTML + sentence count. `POST /upload/ref` saves the WAV, triggers ElevenLabs transcription, and returns the transcript. |
| **8 — WebSocket `/ws`** | Accepts one persistent connection per page load. Dispatches `play` messages by spinning a daemon thread running `_generate` and relaying its output messages back over the socket. `pause` and `resume` toggle `_playback_event`. `stop` sets `_stop_flag` and unblocks `_playback_event` so the producer thread can exit cleanly. |
| **9 — Generation `_generate`** | Validates engine/document/voice state, then launches a **producer thread** that iterates sentence pairs (2 sentences per chunk), calls the selected TTS function, writes each pair to `pair_NNNN.wav`, and puts it on a bounded `queue.Queue(maxsize=2)`. The main thread (consumer) reads from the queue, concatenates all parts so far into a growing `session_NNNN.wav`, deletes the previous session file, and sends three WebSocket messages per chunk: `audio_update` (new URL + duration), `highlight` (sentence indices + progress %), and `status` (human-readable pair counter). Between chunks the consumer sleeps for the pair's audio duration while printing live position to the terminal. On stop or completion, the latest session file is copied to `session_final.wav`. |
| **10 — Entry point** | `uvicorn.run(server, host="0.0.0.0", port=7860)` |

---

## main.py — Quick single-file test

`main.py` is a standalone script for testing OmniVoice generation directly, without the server or browser.

```bash
python main.py
```

It loads the model from `./model`, generates a French sentence in the voice from `austin.wav`, and writes the result to `clone_out.wav`. Edit the `text` and `ref_audio` arguments at the bottom of the file to test different inputs.

This file is not part of the document reader — it is a development/testing utility.

---

## Output files

```
outputs/
  <doc_stem>_<voice_stem>/
    pair_0000.wav          # individual sentence-pair clips
    pair_0001.wav
    ...
    session_final.wav      # full session — the complete document audio
```

Old session files are deleted as new ones are written to keep disk usage low. Only `session_final.wav` persists after a completed run.
