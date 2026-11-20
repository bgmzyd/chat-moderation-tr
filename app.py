"""Real-Time Chat Moderation - Gradio application (R1, R5).

Every message is labeled live (clean / offensive) with P(offensive) and the threshold, offensive
messages are masked, and a side panel logs the response time of every message.
Default model: the distilled BERTurk student, INT8 on ONNX Runtime (CPU). Other models selectable.

Run:  python app.py            (downloads the weights from the Hugging Face Hub on first run)
"""
import argparse
import html
import inspect
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # the app runs on CPU

import gradio as gr

from src import config as C
from src.hub import download, models_ready
from src.models import LexiconPredictor, load_model
from src.normalize import normalize, tokenize

DEFAULT = C.APP_MODEL
CHOICES = [(C.MODEL_NAMES[k], k) for k in C.MODEL_KEYS]
_cache = {}


def get_model(key):
    if key not in _cache:
        m = load_model(key)
        for _ in range(3):              # warm-up, so the log never shows a cold first call
            m.predict_one("merhaba nasılsın")
        _cache[key] = m
    return _cache[key]


def classify(key, text):
    m = get_model(key)
    t0 = time.perf_counter()
    p, y = m.predict_one(text)                # same code path as src/evaluate.py
    ms = (time.perf_counter() - t0) * 1000
    return y, m.threshold, p, ms


def mask(text):
    """Mask offensive words found by the edit-distance lexicon (s****); if none is found
    the whole message is hidden."""
    lex = get_model("lexicon").m
    words, hit = text.split(), False
    for i, w in enumerate(words):
        toks = [t for t in tokenize(normalize(w)) if not t.startswith("<")]
        if any(lex.match_token(t) for t in toks):
            words[i], hit = w[0] + "*" * (len(w) - 1), True
    return " ".join(words) if hit else "▇▇▇▇▇▇ (message hidden)"


def bubble(text, y, thr, p, ms, model_key):
    ok = ms <= C.LATENCY_BUDGET_MS
    lat = f"<span style='color:{'#15803d' if ok else '#b91c1c'}'>{ms:.1f} ms</span>"
    meta = f"<div class='meta'>{C.MODEL_NAMES[model_key]} · {lat}</div>"
    if y:
        return (f"<div class='msg off'><span class='tag off'>OFFENSIVE · P={p:.2f} ≥ {thr:.2f}</span> "
                f"{html.escape(mask(text))}<details><summary>show original</summary>"
                f"{html.escape(text)}</details>{meta}</div>")
    return (f"<div class='msg ok'><span class='tag ok'>CLEAN · P={p:.2f} &lt; {thr:.2f}</span> "
            f"{html.escape(text)}{meta}</div>")


def render_chat(items):
    # newest first in the DOM + flex column-reverse = chat anchored at the bottom
    return "<div class='chat'>" + ("".join(reversed(items)) or
            "<div class='empty'>Type a message below - it is moderated live.</div>") + "</div>"


def summary(log):
    if not log:
        return f"Latency budget: **{C.LATENCY_BUDGET_MS:.0f} ms** per message on CPU"
    ms = [r[4] for r in log]
    within = sum(m <= C.LATENCY_BUDGET_MS for m in ms)
    return (f"Latency budget: **{C.LATENCY_BUDGET_MS:.0f} ms** · messages: {len(ms)} · "
            f"mean: **{sum(ms) / len(ms):.1f} ms** · within budget: {within}/{len(ms)}")


def on_send(text, model_key, compare, chat, log):
    text = (text or "").strip()
    if not text:
        return gr.update(), render_chat(chat), log, summary(log), gr.update(), chat, log
    y, thr, p, ms = classify(model_key, text)
    chat = chat + [bubble(text, y, thr, p, ms, model_key)]
    log = [[time.strftime("%H:%M:%S"), C.MODEL_NAMES[model_key].split(":", 1)[-1].strip(),
            "OFFENSIVE" if y else "clean", round(p, 3), round(ms, 2),
            "✓" if ms <= C.LATENCY_BUDGET_MS else "✗", text[:40]]] + log
    table = gr.update()
    if compare:
        rows = []
        for k in C.MODEL_KEYS:
            yk, tk, pk, mk = classify(k, text)
            rows.append([C.MODEL_NAMES[k], "OFFENSIVE" if yk else "clean", round(pk, 3),
                         round(tk, 2), round(mk, 2)])
        table = gr.update(value=rows, visible=True)
    return "", render_chat(chat), log, summary(log), table, chat, log


CSS = """
.chat{height:430px;overflow-y:auto;display:flex;flex-direction:column-reverse;gap:8px;padding:10px;
      border:1px solid #e5e7eb;border-radius:10px;background:#fafafa}
.msg{padding:8px 12px;border-radius:10px;max-width:92%;color:#111;line-height:1.4}
.msg.ok{background:#ecfdf5;border:1px solid #a7f3d0}
.msg.off{background:#fef2f2;border:1px solid #fecaca}
.tag{font-size:11px;font-weight:700;padding:1px 6px;border-radius:6px;margin-right:6px}
.tag.ok{background:#bbf7d0;color:#065f46}.tag.off{background:#fecaca;color:#991b1b}
.meta{font-size:11px;color:#6b7280;margin-top:3px}
details{font-size:12px;color:#6b7280}.empty{color:#9ca3af;margin:auto}
"""

EXAMPLES = ["Maç harikaydı, tebrikler herkese!", "s4l4k misin sen ya",
            "Bu hakem tam bir a.p.t.a.l", "Oyunu kaybettik ama çok eğlendik 😄",
            "Seni sevmiyorum ama saygı duyuyorum", "Haddini bil şerefsizzz"]


# Gradio >= 6 takes css in launch(), older versions in Blocks()
_CSS_IN_LAUNCH = "css" in inspect.signature(gr.Blocks.launch).parameters


def build():
    kw = {} if _CSS_IN_LAUNCH else {"css": CSS}
    with gr.Blocks(title="Real-Time Chat Moderation (TR)", **kw) as demo:
        gr.Markdown("## 🎮 Real-Time Chat Moderation · Turkish\n"
                    "Each message is classified **clean / offensive** live on CPU: offensive when "
                    "P(offensive) ≥ the model's threshold (tuned on validation). "
                    "Default model: **distilled BERTurk student, INT8, ONNX Runtime**.")
        chat_state, log_state = gr.State([]), gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.HTML(render_chat([]))
                with gr.Row():
                    box = gr.Textbox(placeholder="Type a chat message and press Enter",
                                     show_label=False, scale=5)
                    send = gr.Button("Send", variant="primary", scale=1)
                gr.Examples(EXAMPLES, inputs=box, label="Try (click, then Send)")
            with gr.Column(scale=2):
                model = gr.Dropdown(CHOICES, value=DEFAULT, label="Model")
                compare = gr.Checkbox(False, label="Also run all models side by side")
                stats = gr.Markdown(summary([]))
                log = gr.Dataframe(headers=["time", "model", "label", "P(off)", "ms", "≤ 50 ms", "message"],
                                   value=[], label="Response-time log", interactive=False,
                                   wrap=True)
                cmp = gr.Dataframe(headers=["model", "label", "P(offensive)", "threshold", "ms"], visible=False,
                                   label="All models on the last message", interactive=False)
        ins = [box, model, compare, chat_state, log_state]
        outs = [box, chat, log, stats, cmp, chat_state, log_state]
        box.submit(on_send, ins, outs)
        send.click(on_send, ins, outs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="public gradio link (e.g. on Colab)")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    if not models_ready():
        download()
    for k in ("lexicon", DEFAULT):      # load + warm up before the UI opens
        get_model(k)
    kw = {"css": CSS} if _CSS_IN_LAUNCH else {}
    build().launch(share=a.share, server_port=a.port, **kw)
