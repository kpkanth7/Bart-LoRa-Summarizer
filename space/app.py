import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE_MODEL = "facebook/bart-large-cnn"
ADAPTER_PATH = "./lora_adapter_bart_large_cnn"

# Load tokenizer once
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load baseline model (no LoRA)
baseline_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

# Load LoRA model (base + adapter)
lora_base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
lora_model = PeftModel.from_pretrained(lora_base, ADAPTER_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
baseline_model.to(device).eval()
lora_model.to(device).eval()

def _generate_summary(model, text, max_new_tokens=80, min_new_tokens=25, num_beams=6):
    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=int(min_new_tokens),
            num_beams=int(num_beams),
            length_penalty=2.5,
            no_repeat_ngram_size=3,
            early_stopping=True
        )

    return tokenizer.decode(out_ids[0], skip_special_tokens=True)

def compare_summaries(text, max_new_tokens=80, num_beams=6):
    text = (text or "").strip()
    if len(text) == 0:
        return "Please paste some text.", "Please paste some text."

    # Friendly guardrail: short inputs often look like copy/paste “summaries”
    if len(text.split()) < 60:
        msg = "Tip: paste at least ~6–8 sentences (60+ words) to see stronger summarization."
        return msg, msg

    baseline = _generate_summary(
        baseline_model,
        text,
        max_new_tokens=max_new_tokens,
        min_new_tokens=25,
        num_beams=num_beams
    )

    lora = _generate_summary(
        lora_model,
        text,
        max_new_tokens=max_new_tokens,
        min_new_tokens=25,
        num_beams=num_beams
    )

    return baseline, lora

with gr.Blocks() as demo:
    gr.Markdown(
        """
# BART Summarizer — Baseline vs LoRA Fine-Tuned

Paste an article/paragraph and compare:
- **Baseline**: `facebook/bart-large-cnn`
- **LoRA Fine-tuned**: adapter trained on CNN/DailyMail

"""
    )

    inp = gr.Textbox(
        lines=10,
        label="Paste text to summarize",
        placeholder="Paste a paragraph or article (recommended: 6–10 sentences)..."
    )

    with gr.Row():
        max_new = gr.Slider(32, 160, value=80, step=8, label="Max new tokens (shorter vs longer)")
        beams = gr.Slider(1, 8, value=6, step=1, label="Beams (quality vs speed)")

    btn = gr.Button("Summarize")

    with gr.Row():
        out_base = gr.Textbox(lines=8, label="Baseline Summary (BART)")
        out_lora = gr.Textbox(lines=8, label="LoRA Summary (Fine-tuned)")

    btn.click(
        fn=compare_summaries,
        inputs=[inp, max_new, beams],
        outputs=[out_base, out_lora]
    )

demo.launch()