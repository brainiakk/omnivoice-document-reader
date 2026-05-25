from io import BytesIO
import asyncio, json, os, queue, re, html as _html, shutil, tempfile, threading, time
from dataclasses import dataclass, field
from pathlib import Path

import nltk
import numpy as np
import soundfile as sf
import torch
import uvicorn
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ─── 1. MODEL ─────────────────────────────────────────────────────────
print("=" * 60)
print("Loading OmniVoice…")
from omnivoice import OmniVoice
try:
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
    _lang_ok = True
except ImportError:
    _lang_ok = False

omni_tts = OmniVoice.from_pretrained(
    "./model", local_files_only=True,
    device_map="mps", dtype=torch.bfloat16, load_asr=False,
)
print("OmniVoice ready.")

print("Loading Supertonic 3…")
try:
    from supertonic import TTS as _SupertonicTTS
    st_tts = _SupertonicTTS(model_dir="./supertonic3")
    print("Supertonic 3 ready.")
except Exception as _st_err:
    st_tts = None
    print(f"Supertonic 3 failed to load: {_st_err}")
print("=" * 60)

OMNI_SR = 24_000
ST_SR   = 44_100   # from supertonic3/onnx/tts.json
nltk.download("punkt_tab", quiet=True)

if _lang_ok:
    try:
        LANG_CHOICES = sorted(
            [(lang_display_name(c), c) for c in LANG_NAMES], key=lambda x: x[0]
        )
    except Exception:
        LANG_CHOICES = sorted([(c, c) for c in LANG_NAMES])
else:
    LANG_CHOICES = [
        ("English", "en"), ("Spanish", "es"), ("French", "fr"),
        ("German", "de"), ("Chinese", "zh"), ("Japanese", "ja"),
    ]

# ─── 2. DOCUMENT PARSING ──────────────────────────────────────────────
@dataclass
class DocBlock:
    kind: str
    raw_text: str
    sentences: list[tuple[int, str]] = field(default_factory=list)


def _sanitize(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:48]


def parse_pdf(path: str) -> list[DocBlock]:
    import fitz
    doc = fitz.open(path)
    blocks: list[DocBlock] = []
    buf: list[str] = []

    def flush():
        t = " ".join(buf).strip()
        if t:
            blocks.append(DocBlock("p", t))
        buf.clear()

    for page in doc:
        for b in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]:
            if b.get("type") != 0:
                continue
            for line in b["lines"]:
                text, msz = "", 0
                for span in line["spans"]:
                    text += span["text"]
                    msz = max(msz, span["size"])
                text = text.strip()
                if not text:
                    flush(); continue
                if msz >= 18:
                    flush(); blocks.append(DocBlock("h1", text))
                elif msz >= 14:
                    flush(); blocks.append(DocBlock("h2", text))
                elif msz >= 12:
                    flush(); blocks.append(DocBlock("h3", text))
                else:
                    buf.append(text)
        flush()
    return blocks


def parse_docx(path: str) -> list[DocBlock]:
    from docx import Document
    blocks: list[DocBlock] = []
    for para in Document(path).paragraphs:
        t = para.text.strip()
        if not t:
            continue
        s = para.style.name if para.style else ""
        kind = ("h1" if "Heading 1" in s else "h2" if "Heading 2" in s
                else "h3" if "Heading 3" in s or "Heading 4" in s else "p")
        blocks.append(DocBlock(kind, t))
    return blocks


def parse_markdown(path: str) -> list[DocBlock]:
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    blocks: list[DocBlock] = []
    buf: list[str] = []
    in_code = False

    def flush():
        t = " ".join(buf).strip()
        t = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", t)
        t = re.sub(r"`(.+?)`", r"\1", t)
        t = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", t).strip()
        if t:
            blocks.append(DocBlock("p", t))
        buf.clear()

    for line in lines:
        s = line.rstrip()
        if s.startswith("```"):
            in_code = not in_code; flush(); continue
        if in_code:
            continue
        if not s:
            flush(); continue
        if m := re.match(r"^(#{1,6})\s+(.*)", s):
            flush()
            lvl = len(m.group(1))
            blocks.append(DocBlock(
                "h1" if lvl == 1 else "h2" if lvl == 2 else "h3",
                m.group(2).strip()
            ))
        else:
            buf.append(s)
    flush()
    return blocks


def parse_document(path: str) -> list[DocBlock]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext in (".md", ".txt", ".markdown"):
        return parse_markdown(path)
    raise ValueError(f"Unsupported: {ext}")


def assign_sentences(blocks: list[DocBlock]) -> tuple[list[DocBlock], list[str]]:
    flat: list[str] = []
    for block in blocks:
        if block.kind.startswith("h"):
            continue
        for s in nltk.sent_tokenize(block.raw_text):
            s = s.strip()
            if len(s) > 10:
                block.sentences.append((len(flat), s))
                flat.append(s)
    return blocks, flat


def build_doc_html(blocks: list[DocBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        safe = _html.escape(block.raw_text)
        if block.kind == "h1":
            parts.append(f'<div class="doc-h1">{safe}</div>')
        elif block.kind == "h2":
            parts.append(f'<div class="doc-h2">{safe}</div>')
        elif block.kind == "h3":
            parts.append(f'<div class="doc-h3">{safe}</div>')
        else:
            if not block.sentences:
                parts.append(f'<p class="doc-para">{safe}</p>')
            else:
                spans = " ".join(
                    f'<span id="s-{i}" class="sentence">{_html.escape(s)}</span>'
                    for i, s in block.sentences
                )
                parts.append(f'<p class="doc-para">{spans}</p>')
    return "".join(parts)


# ─── 3. AUDIO GENERATION ──────────────────────────────────────────────
def _normalise(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0] if arr.shape[0] == 1 else arr[:, 0]
    arr = arr.squeeze()
    if arr.size > 0 and np.abs(arr).max() > 1.5:
        arr = arr / 32768.0
    return arr


def gen_omni(text, ref_audio, ref_text, language, voice_design, design_prompt) -> np.ndarray:
    print(f"[gen_omni] {text[:60]!r}  lang={language}  design={voice_design}")
    if voice_design and design_prompt.strip():
        raw = omni_tts.generate(text=text, instruct=design_prompt.strip(), language=language)
    else:
        kwargs = {"text": text, "ref_audio": ref_audio, "language": language}
        if ref_text.strip():
            kwargs["ref_text"] = ref_text.strip()
        raw = omni_tts.generate(**kwargs)
    audio = raw[0] if isinstance(raw, (list, tuple)) else raw
    return _normalise(audio)


def gen_supertonic(text: str, voice: str, lang: str) -> np.ndarray:
    if st_tts is None:
        raise RuntimeError("Supertonic 3 failed to load at startup.")
    print(f"[gen_supertonic] {text[:60]!r}  voice={voice}  lang={lang}")
    style = st_tts.get_voice_style(voice)
    wav, _ = st_tts.synthesize(text, voice_style=style, lang=lang or None)
    return _normalise(wav)


def transcribe_elevenlabs(audio_path: str) -> str:
    client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    with open(audio_path, "rb") as f:
        data = f.read()
    t = client.speech_to_text.convert(
        file=BytesIO(data), model_id="scribe_v2", language_code="eng"
    )
    return t.text


# ─── 4. SESSION STATE ─────────────────────────────────────────────────
_sentences:      list[str]      = []
_doc_blocks:     list[DocBlock] = []
_doc_path:       str | None     = None
_ref_audio_path: str | None     = None
_ref_text:       str            = ""
_stop_flag       = threading.Event()
_playback_event  = threading.Event()
_playback_event.set()

# ─── 5. FASTAPI APP ───────────────────────────────────────────────────
server = FastAPI()
Path("outputs").mkdir(exist_ok=True)
server.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ─── 6. HTML PAGE ─────────────────────────────────────────────────────
_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Document Reader</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f0f0f;color:#d4d4d4;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#hdr{padding:.75rem 1.25rem;border-bottom:1px solid #222;flex-shrink:0}
#hdr h1{font-size:1.05rem;font-weight:600;color:#f0f4f8}
#hdr p{font-size:.75rem;color:#666;margin-top:2px}
#body{display:flex;flex:1;overflow:hidden}
/* sidebar */
#sb{width:260px;min-width:260px;border-right:1px solid #222;overflow-y:auto;padding:.85rem;display:flex;flex-direction:column;gap:.65rem}
.lbl{font-size:.7rem;font-weight:600;color:#666;text-transform:uppercase;letter-spacing:.05em;margin-bottom:2px}
.faux-btn{display:block;width:100%;padding:.38rem .6rem;background:#1c1c1c;border:1px solid #333;border-radius:5px;color:#ccc;cursor:pointer;font-size:.82rem;text-align:center;transition:border-color .15s}
.faux-btn:hover{border-color:#555}
input[type=file]{display:none}
select,textarea{width:100%;padding:.38rem .55rem;background:#1c1c1c;border:1px solid #333;border-radius:5px;color:#d4d4d4;font-size:.82rem;outline:none}
select:focus,textarea:focus{border-color:#4a90d9}
textarea{resize:vertical;min-height:52px}
.ck-row{display:flex;align-items:center;gap:.45rem;font-size:.82rem;cursor:pointer}
.ck-row input{cursor:pointer;accent-color:#4a90d9}
#status{background:#161616;border:1px solid #222;border-radius:5px;padding:.45rem .55rem;font-size:.78rem;color:#777;min-height:38px;line-height:1.4}
#play-btn{width:100%;padding:.48rem;background:#1d4ed8;border:none;border-radius:5px;color:#fff;font-weight:600;cursor:pointer;font-size:.88rem;transition:background .15s}
#play-btn:hover{background:#1e40af}
#play-btn:disabled{background:#1e3a5f;color:#555;cursor:not-allowed}
.btn-row{display:flex;gap:.45rem}
.sm-btn{flex:1;padding:.38rem;background:#1c1c1c;border:1px solid #333;border-radius:5px;color:#ccc;cursor:pointer;font-size:.82rem;transition:border-color .15s}
.sm-btn:hover{border-color:#555}
#stop-btn{color:#f87171;border-color:#7f1d1d}
#stop-btn:hover{background:#1a0808;border-color:#f87171}
/* doc area */
#doc-area{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:.85rem;gap:.65rem}
#prog-wrap{height:4px;background:#222;border-radius:3px;overflow:hidden;flex-shrink:0}
#prog-fill{height:100%;width:0%;background:linear-gradient(90deg,#63b3ed,#a78bfa);transition:width .35s ease}
#doc-view{flex:1;overflow-y:auto;background:#161616;border-radius:7px;padding:1.4rem 1.8rem;font-family:Georgia,serif;font-size:.98rem;line-height:1.85;color:#ccc}
#doc-view.empty{color:#444;display:flex;align-items:center;justify-content:center}
.doc-h1{font-size:1.55rem;font-weight:700;color:#f0f4f8;margin:1.5rem 0 .45rem;border-bottom:1px solid #272727;padding-bottom:.3rem}
.doc-h2{font-size:1.25rem;font-weight:600;color:#cbd5e0;margin:1.2rem 0 .38rem}
.doc-h3{font-size:1.05rem;font-weight:500;color:#a0aec0;margin:.9rem 0 .28rem;font-style:italic}
.doc-para{margin:.45rem 0}
.sentence{display:inline;border-radius:3px;padding:1px 3px;transition:background .2s,color .2s}
.sentence.active{background:rgba(99,179,237,.22);color:#eff6ff;box-shadow:0 0 0 1px rgba(99,179,237,.35)}
.sentence.done{color:#444}
/* player */
#player-wrap{background:#111;border-radius:7px;padding:.65rem;flex-shrink:0}
#player-lbl{font-size:.72rem;color:#555;margin-bottom:5px}
#player{width:100%;display:block}
#player-status{font-size:.68rem;color:#444;margin-top:4px}
</style>
</head>
<body>
<div id="hdr">
  <h1>Document Reader</h1>
  <p>Drop in a document · upload a voice · hit Play · fully local</p>
</div>
<div id="body">
  <div id="sb">
    <div>
      <div class="lbl">🤖 TTS Engine</div>
      <select id="engine-sel">
        <option value="omnivoice">OmniVoice (Voice Cloning)</option>
        <option value="supertonic">Supertonic (Fast TTS)</option>
      </select>
    </div>
    <div id="omnivoice-col">
      <div class="lbl" style="margin-top:4px">🎙 Voice</div>
      <label class="ck-row"><input type="checkbox" id="vd-toggle"> Voice Design Mode</label>
      <div id="clone-sec" style="margin-top:6px">
        <div class="lbl">Reference WAV (3–10 sec)</div>
        <label class="faux-btn" for="ref-input">📂 Upload WAV</label>
        <input type="file" id="ref-input" accept=".wav,.mp3,.m4a,.ogg">
        <textarea id="ref-tx" placeholder="Reference transcript (auto-filled)" rows="2" style="margin-top:4px"></textarea>
      </div>
      <div id="design-sec" style="display:none;margin-top:6px">
        <div class="lbl">Voice Design Prompt</div>
        <textarea id="design-prompt" placeholder="e.g. Male, deep baritone, American accent" rows="3"></textarea>
      </div>
      <div style="margin-top:6px">
        <div class="lbl">🌍 Language (600+)</div>
        <select id="lang-sel"></select>
      </div>
    </div>
    <div id="supertonic-col" style="display:none">
      <div class="lbl" style="margin-top:4px">🔊 Speaker</div>
      <select id="st-voice-sel">
        <option value="M1">Male 1</option>
        <option value="M2">Male 2</option>
        <option value="M3">Male 3</option>
        <option value="M4">Male 4</option>
        <option value="F1">Female 1</option>
        <option value="F2">Female 2</option>
        <option value="F3">Female 3</option>
        <option value="F4">Female 4</option>
        <option value="F5">Female 5</option>
      </select>
      <div class="lbl" style="margin-top:6px">🌍 Language</div>
      <select id="st-lang-sel">
        <option value="en">English</option>
        <option value="ar">Arabic</option>
        <option value="bg">Bulgarian</option>
        <option value="cs">Czech</option>
        <option value="da">Danish</option>
        <option value="de">German</option>
        <option value="el">Greek</option>
        <option value="es">Spanish</option>
        <option value="et">Estonian</option>
        <option value="fi">Finnish</option>
        <option value="fr">French</option>
        <option value="hi">Hindi</option>
        <option value="hr">Croatian</option>
        <option value="hu">Hungarian</option>
        <option value="id">Indonesian</option>
        <option value="it">Italian</option>
        <option value="ja">Japanese</option>
        <option value="ko">Korean</option>
        <option value="lt">Lithuanian</option>
        <option value="lv">Latvian</option>
        <option value="nl">Dutch</option>
        <option value="pl">Polish</option>
        <option value="pt">Portuguese</option>
        <option value="ro">Romanian</option>
        <option value="ru">Russian</option>
        <option value="sk">Slovak</option>
        <option value="sl">Slovenian</option>
        <option value="sv">Swedish</option>
        <option value="tr">Turkish</option>
        <option value="uk">Ukrainian</option>
        <option value="vi">Vietnamese</option>
      </select>
    </div>
    <div>
      <div class="lbl">📄 Document</div>
      <label class="faux-btn" for="doc-input">📂 Upload PDF / DOCX / MD / TXT</label>
      <input type="file" id="doc-input" accept=".pdf,.docx,.md,.txt,.markdown">
    </div>
    <div>
      <div class="lbl">▶ Playback</div>
      <button id="play-btn">▶&nbsp; Play</button>
      <div class="btn-row" style="margin-top:5px">
        <button class="sm-btn" id="pause-btn">⏸&nbsp; Pause</button>
        <button class="sm-btn" id="stop-btn">⏹&nbsp; Stop</button>
      </div>
    </div>
    <div id="status">Load a document to begin.</div>
  </div>

  <div id="doc-area">
    <div id="prog-wrap"><div id="prog-fill"></div></div>
    <div id="doc-view" class="empty">No document loaded.</div>
    <div id="player-wrap">
      <div id="player-lbl">📼 Session audio — grows as generation progresses</div>
      <audio id="player" controls></audio>
      <div id="player-status">No audio yet — press Play to begin</div>
    </div>
  </div>
</div>

<script>
// ── Language select ───────────────────────────────────────────────────
(function(){
  const choices = LANG_CHOICES_PLACEHOLDER;
  const sel = document.getElementById('lang-sel');
  choices.forEach(([name, code]) => {
    const o = document.createElement('option');
    o.value = code; o.textContent = name;
    if (code === 'en') o.selected = true;
    sel.appendChild(o);
  });
})();

// ── Voice design toggle ───────────────────────────────────────────────
document.getElementById('engine-sel').addEventListener('change', function(){
  const isOmni = this.value === 'omnivoice';
  document.getElementById('omnivoice-col').style.display  = isOmni ? '' : 'none';
  document.getElementById('supertonic-col').style.display = isOmni ? 'none' : '';
});

document.getElementById('vd-toggle').addEventListener('change', function(){
  document.getElementById('clone-sec').style.display  = this.checked ? 'none' : '';
  document.getElementById('design-sec').style.display = this.checked ? ''     : 'none';
});

// ── Document upload ───────────────────────────────────────────────────
document.getElementById('doc-input').addEventListener('change', async function(){
  const f = this.files[0]; if (!f) return;
  setStatus('Parsing document…');
  const fd = new FormData(); fd.append('file', f);
  try {
    const r   = await fetch('/upload/doc', {method:'POST', body:fd});
    const dat = await r.json();
    const dv  = document.getElementById('doc-view');
    dv.innerHTML = dat.html;
    dv.classList.remove('empty');
    setStatus(`${dat.count} sentences loaded — ready.`);
  } catch(e) { setStatus('Parse error: ' + e.message); }
});

// ── Reference audio upload ────────────────────────────────────────────
document.getElementById('ref-input').addEventListener('change', async function(){
  const f = this.files[0]; if (!f) return;
  setStatus('Transcribing reference audio…');
  const fd = new FormData(); fd.append('file', f);
  try {
    const r   = await fetch('/upload/ref', {method:'POST', body:fd});
    const dat = await r.json();
    document.getElementById('ref-tx').value = dat.transcript || '';
    setStatus('Voice loaded — press Play.');
  } catch(e) { setStatus('Transcription error: ' + e.message); }
});

// ── Audio player ──────────────────────────────────────────────────────
const player      = document.getElementById('player');
let playerReady = false;  // has ever loaded audio
let wasPlaying  = false;  // currently playing right now
let userPaused  = false;  // user explicitly clicked Pause (not just audio ending)

player.addEventListener('play',  () => { wasPlaying = true; });
player.addEventListener('pause', () => {
  // Only mark userPaused when the audio didn't just reach its natural end.
  // When audio ends the browser fires 'ended' then 'pause'; we ignore that pair.
  if (!player.ended) wasPlaying = false;
});
player.addEventListener('ended', () => { wasPlaying = false; });

function loadAudio(url, duration) {
  const savedTime = player.currentTime || 0;
  const didEnd    = player.ended;   // capture before src swap resets it

  // Resume if:  was mid-play when audio grew  |  audio just ended naturally  |  first ever load
  // Don't resume if: user explicitly clicked Pause
  const shouldPlay = !userPaused && (wasPlaying || didEnd || !playerReady);

  player.src = url;

  player.addEventListener('canplay', function handler() {
    player.removeEventListener('canplay', handler);
    if (savedTime > 0.1 && savedTime < player.duration - 0.05) {
      try { player.fastSeek ? player.fastSeek(savedTime) : (player.currentTime = savedTime); }
      catch(e) { player.currentTime = savedTime; }
    }
    if (shouldPlay) {
      player.play().catch(e => console.warn('play():', e));
      wasPlaying = true;
    }
    playerReady = true;
    if (duration > 0)
      document.getElementById('player-status').textContent =
        'Session ≈ ' + duration.toFixed(1) + 's';
  });
}

// ── WebSocket ─────────────────────────────────────────────────────────
const ws = new WebSocket('ws://' + location.host + '/ws');
ws.onopen  = () => console.log('WS connected');
ws.onerror = e  => console.warn('WS error', e);
ws.onclose = () => console.log('WS closed');

ws.onmessage = function(e) {
  const msg = JSON.parse(e.data);
  switch (msg.type) {

    case 'audio_update':
      loadAudio(msg.url, msg.duration);
      break;

    case 'highlight':
      document.querySelectorAll('.sentence.active').forEach(el => {
        el.classList.remove('active'); el.classList.add('done');
      });
      msg.indices.forEach((idx, n) => {
        const el = document.getElementById('s-' + idx);
        if (!el) return;
        el.classList.add('active');
        if (n === 0) el.scrollIntoView({behavior:'smooth', block:'center'});
      });
      const bar = document.getElementById('prog-fill');
      if (bar) bar.style.width = msg.pct + '%';
      break;

    case 'status':
      setStatus(msg.text);
      break;

    case 'done':
      document.getElementById('play-btn').disabled = false;
      document.getElementById('play-btn').textContent = '▶  Play';
      document.getElementById('pause-btn').textContent = '⏸  Pause';
      if (msg.url) loadAudio(msg.url, 0);
      setStatus(msg.stopped ? 'Stopped.' : 'Done! Full session saved.');
      break;

    case 'error':
      setStatus('Error: ' + msg.text);
      document.getElementById('play-btn').disabled = false;
      document.getElementById('play-btn').textContent = '▶  Play';
      break;
  }
};

// ── Playback controls ─────────────────────────────────────────────────
document.getElementById('play-btn').addEventListener('click', function() {
  const vd = document.getElementById('vd-toggle').checked;
  playerReady = false;
  wasPlaying  = false;
  userPaused  = false;
  this.disabled    = true;
  this.textContent = '⏸  Playing';
  document.getElementById('pause-btn').textContent = '⏸  Pause';
  const eng = document.getElementById('engine-sel').value;
  ws.send(JSON.stringify({
    type:          'play',
    engine:        eng,
    language:      eng === 'omnivoice'
                     ? document.getElementById('lang-sel').value
                     : document.getElementById('st-lang-sel').value,
    voice_design:  vd,
    design_prompt: document.getElementById('design-prompt').value.trim(),
    ref_text:      document.getElementById('ref-tx').value.trim(),
    st_voice:      document.getElementById('st-voice-sel').value,
  }));
});

document.getElementById('pause-btn').addEventListener('click', function() {
  if (this.textContent.includes('Pause')) {
    userPaused = true;
    player.pause();
    this.textContent = '▶  Resume';
    ws.send(JSON.stringify({type:'pause'}));
  } else {
    userPaused = false;
    player.play().catch(e => {});
    this.textContent = '⏸  Pause';
    ws.send(JSON.stringify({type:'resume'}));
  }
});

document.getElementById('stop-btn').addEventListener('click', function() {
  userPaused = false;
  player.pause();
  ws.send(JSON.stringify({type:'stop'}));
  const pb = document.getElementById('play-btn');
  pb.disabled = false;
  pb.textContent = '▶  Play';
  document.getElementById('pause-btn').textContent = '⏸  Pause';
});

function setStatus(t) {
  document.getElementById('status').textContent = t;
}
</script>
</body>
</html>"""

# Inject language choices at startup
_HTML_PAGE = _HTML_TEMPLATE.replace(
    "LANG_CHOICES_PLACEHOLDER", json.dumps(LANG_CHOICES)
)


# ─── 7. ROUTES ────────────────────────────────────────────────────────
@server.get("/")
async def index():
    return HTMLResponse(_HTML_PAGE)


@server.post("/upload/doc")
async def upload_doc(file: UploadFile = File(...)):
    global _sentences, _doc_blocks, _doc_path
    contents = await file.read()
    suffix = Path(file.filename or "upload").suffix.lower() or ".txt"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    Path(tmp).write_bytes(contents)
    try:
        blocks = parse_document(tmp)
        blocks, sentences = assign_sentences(blocks)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    _doc_blocks = blocks
    _sentences  = sentences
    _doc_path   = tmp
    return JSONResponse({"html": build_doc_html(blocks), "count": len(sentences)})


@server.post("/upload/ref")
async def upload_ref(file: UploadFile = File(...)):
    global _ref_audio_path, _ref_text
    contents = await file.read()
    fd, tmp = tempfile.mkstemp(suffix=Path(file.filename or "ref.wav").suffix or ".wav")
    os.close(fd)
    Path(tmp).write_bytes(contents)
    _ref_audio_path = tmp
    try:
        _ref_text = transcribe_elevenlabs(tmp)
    except Exception:
        _ref_text = ""
    return JSONResponse({"transcript": _ref_text})


# ─── 8. WEBSOCKET ─────────────────────────────────────────────────────
@server.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _stop_flag, _playback_event
    await ws.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")
            if t == "play":
                _stop_flag.clear()
                _playback_event.set()
                msg_q: asyncio.Queue = asyncio.Queue()

                def put(msg):
                    asyncio.run_coroutine_threadsafe(msg_q.put(msg), loop)

                def run_gen():
                    try:
                        _generate(data, put)
                    except Exception as ex:
                        put({"type": "error", "text": str(ex)})
                    finally:
                        put(None)

                threading.Thread(target=run_gen, daemon=True).start()

                while True:
                    msg = await msg_q.get()
                    if msg is None:
                        break
                    try:
                        await ws.send_json(msg)
                    except Exception:
                        break

            elif t == "pause":
                _playback_event.clear()
            elif t == "resume":
                _playback_event.set()
            elif t == "stop":
                _stop_flag.set()
                _playback_event.set()

    except WebSocketDisconnect:
        _stop_flag.set()
        _playback_event.set()


# ─── 9. GENERATION ────────────────────────────────────────────────────
def _generate(data: dict, send):
    engine        = data.get("engine", "omnivoice")
    lang          = data.get("language", "en")
    voice_design  = bool(data.get("voice_design", False))
    design_prompt = data.get("design_prompt", "")
    ref_text      = data.get("ref_text", "") or _ref_text
    st_voice      = data.get("st_voice", "M1")

    if engine == "supertonic" and st_tts is None:
        send({"type": "error", "text": "Supertonic 3 failed to load at startup — check terminal."}); return

    if not _sentences:
        send({"type": "error", "text": "Load a document first."}); return
    if engine == "omnivoice":
        if not voice_design and not _ref_audio_path:
            send({"type": "error", "text": "Upload a reference WAV first."}); return
        if voice_design and not design_prompt.strip():
            send({"type": "error", "text": "Enter a voice design prompt first."}); return

    sr = OMNI_SR if engine == "omnivoice" else ST_SR
    total  = len(_sentences)
    pairs  = [list(range(i, min(i + 2, total))) for i in range(0, total, 2)]
    n_pairs = len(pairs)

    doc_stem = _sanitize(Path(_doc_path).stem) if _doc_path else "doc"
    suffix   = "voicedesign" if voice_design else (
        _sanitize(Path(_ref_audio_path).stem) if _ref_audio_path else "ref"
    )
    out_dir = Path("outputs") / f"{doc_stem}_{suffix}"
    if out_dir.exists():
        for f in out_dir.iterdir():
            try: f.unlink()
            except OSError: pass
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_q: queue.Queue = queue.Queue(maxsize=2)

    def producer():
        for p_idx, idx_list in enumerate(pairs):
            if _stop_flag.is_set(): break
            _playback_event.wait()
            if _stop_flag.is_set(): break
            text = " ".join(_sentences[i] for i in idx_list)
            try:
                if engine == "supertonic":
                    arr = gen_supertonic(text, st_voice, lang)
                else:
                    arr = gen_omni(text, _ref_audio_path, ref_text,
                                   lang, voice_design, design_prompt)
                if arr.size == 0: raise ValueError("empty array")
                p_path = str(out_dir / f"pair_{p_idx:04d}.wav")
                sf.write(p_path, arr, sr)
                audio_q.put((p_idx, idx_list, p_path))
            except Exception as ex:
                print(f"[producer] pair {p_idx}: {ex}")
                p_path = str(out_dir / f"pair_{p_idx:04d}.wav")
                sf.write(p_path, np.zeros(int(sr * 0.5), dtype=np.float32), sr)
                audio_q.put((p_idx, idx_list, p_path))
        audio_q.put(None)

    threading.Thread(target=producer, daemon=True).start()

    parts: list[np.ndarray] = []
    session_path:      str | None = None
    prev_session_path: str | None = None
    cumulative_s = 0.0

    while True:
        item = audio_q.get()
        if item is None: break
        p_idx, idx_list, p_path = item

        if _stop_flag.is_set():
            try:
                while True: audio_q.get_nowait()
            except queue.Empty: pass
            break

        chunk, _ = sf.read(p_path, dtype="float32")
        if chunk.ndim > 1: chunk = chunk[:, 0]
        parts.append(chunk)

        full = np.concatenate(parts) if len(parts) > 1 else parts[0]
        new_sess = str(out_dir / f"session_{p_idx:04d}.wav")
        sf.write(new_sess, full, sr)

        if prev_session_path and os.path.exists(prev_session_path):
            try: os.unlink(prev_session_path)
            except OSError: pass

        prev_session_path = session_path
        session_path      = new_sess

        pair_dur    = sf.info(p_path).duration
        session_dur = sf.info(session_path).duration

        # URL served by /outputs StaticFiles mount
        url = f"/outputs/{out_dir.name}/session_{p_idx:04d}.wav"

        send({"type": "audio_update", "url": url, "duration": session_dur})
        send({
            "type":    "highlight",
            "indices": idx_list,
            "total":   total,
            "pct":     int(((max(idx_list) + 1) / total) * 100),
        })
        send({
            "type": "status",
            "text": (f"▶  Pair {p_idx+1}/{n_pairs} · "
                     f"sentences {idx_list[0]+1}–{idx_list[-1]+1} of {total} · "
                     f"session ≈ {session_dur:.1f}s"),
        })

        # Sleep for pair duration, printing live position to terminal
        pair_start = time.time()
        end_t      = pair_start + max(0.0, pair_dur - 0.15)
        while time.time() < end_t:
            if _stop_flag.is_set(): break
            elapsed = time.time() - pair_start
            pos = cumulative_s + min(elapsed, pair_dur)
            print(f"\r[position] {pos:.2f}s / {session_dur:.1f}s", end="", flush=True)
            time.sleep(0.1)
        print()
        cumulative_s += pair_dur

    # Finalise
    stopped = _stop_flag.is_set()
    if session_path and os.path.exists(session_path):
        final = str(out_dir / "session_final.wav")
        shutil.copy2(session_path, final)
        if prev_session_path and os.path.exists(prev_session_path):
            try: os.unlink(prev_session_path)
            except OSError: pass
        if session_path != final and os.path.exists(session_path):
            try: os.unlink(session_path)
            except OSError: pass
        send({
            "type":    "done",
            "url":     f"/outputs/{out_dir.name}/session_final.wav",
            "stopped": stopped,
        })
    else:
        send({"type": "done", "url": None, "stopped": True})


# ─── 10. ENTRY POINT ──────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=7860)
