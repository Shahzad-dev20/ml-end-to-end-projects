"""Gradio interface - one tab per model (Maternal Risk, Pulsar Spotter).

Author: Ahmed Shahzad
Developer: Ahmed Shahzad

Run:  python gradio_app.py      then open http://127.0.0.1:7860
The Flask app (flask_app.py) has the richer interface; this one is the quickest way to try the models.
"""
import inspect

import gradio as gr

from common import get_datasets, predict, warm_up

COLORS = {
    "low risk": "#12B886", "mid risk": "#F5A524", "high risk": "#FF4D6D",
    "not pulsar": "#8F9AC4", "pulsar": "#8C7BFF",
}

CSS = """
.gradio-container { max-width: 1120px !important; margin: 0 auto; }
#hero { background: linear-gradient(135deg, #0B0F26 0%, #1B2160 100%); border-radius: 22px; padding: 28px 32px; color: #E9ECFA; margin-bottom: 8px; }
#hero h1 { margin: 0 0 6px; font-size: 2.2rem; letter-spacing: -0.03em; color: #fff; }
#hero p { margin: 0; color: #B9C0E4; }
.verdict-card { background: #0B0F26; border-radius: 20px; padding: 24px 28px; color: #E9ECFA; }
.verdict-card .small { color: #A3ABD0; font-size: .9rem; margin-bottom: 4px; }
.verdict-card .big { font-size: 2.8rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.05; }
.verdict-card .conf { margin-top: 8px; color: #E9ECFA; }
.verdict-card .warn { margin-top: 12px; background: rgba(245,165,36,.15); border: 1px solid rgba(245,165,36,.5); color: #FFD9A0; border-radius: 10px; padding: 8px 12px; font-size: .85rem; }
.verdict-card .disc { margin-top: 14px; color: #A3ABD0; font-size: .8rem; }
"""


def verdict_html(cfg: dict, out: dict) -> str:
    color = COLORS[out["prediction"]]
    if cfg["task"] == "binary":
        p = next(x["p"] for x in out["probabilities"] if x["label"] == cfg["positive_label"])
        conf = f"Estimated chance this is a real pulsar: <b>{p:.1%}</b>"
    else:
        conf = f"The model is <b>{out['confidence']:.0%}</b> confident in this level."
    warns = "".join(f"<div class='warn'>{w}</div>" for w in out["warnings"])
    return (
        f"<div class='verdict-card'><div class='small'>{cfg['target_label']}</div>"
        f"<div class='big' style='color:{color}'>{out['display']}</div>"
        f"<div class='conf'>{conf}</div>{warns}"
        f"<div class='disc'>{cfg['disclaimer']}</div></div>"
    )


def build_tab(key: str, cfg: dict):
    gr.Markdown(f"{cfg['description']}  \n*Data: {cfg['source']}*")
    names = [f["name"] for f in cfg["features"]]

    with gr.Row():
        with gr.Column(scale=5):
            sliders = []
            for f in cfg["features"]:
                label = f["label"] + (f" ({f['unit']})" if f["unit"] else "")
                sliders.append(
                    gr.Slider(
                        minimum=f["min"], maximum=f["max"], value=f["default"], step=f["step"],
                        label=label, info=f["help"],
                    )
                )
            with gr.Row():
                example_buttons = []
                for ex in cfg["examples"]:
                    example_buttons.append((gr.Button(f"Real example: {ex['display']}", size="sm"), ex["values"]))
                typical = gr.Button("Typical values", size="sm")

        with gr.Column(scale=6):
            card = gr.HTML()
            probs = gr.Label(label="Class probabilities", num_top_classes=len(cfg["classes"]))
            drivers = gr.Dataframe(
                headers=["Input", "Your value", "Effect on the chance (points)"],
                datatype=["str", "str", "number"], interactive=False, label="What moves the result",
            )
            m = cfg["metrics"]
            if key == "maternal":
                gr.Markdown(
                    f"**Model quality (test set of {m['n_test']} women, {m['model_name']})**  \n"
                    f"Accuracy {m['accuracy']:.0%} | Macro F1 {m['macro_f1']:.2f} | High-risk recall {m['high_risk_recall']:.0%}  \n"
                    "*Mid risk is the hardest class to separate.*"
                )
            else:
                gr.Markdown(
                    f"**Model quality (test set of {m['n_test']:,} candidates, {m['model_name']})**  \n"
                    f"Precision {m['precision']:.1%} | Recall {m['recall']:.1%} | F1 {m['f1']:.3f} | PR-AUC {m['pr_auc']:.3f}"
                )

    def run(*vals):
        try:
            out = predict(key, dict(zip(names, vals)))
        except ValueError as err:
            raise gr.Error(str(err))
        table = [
            [d["label"], f"{d['value']:g} {d['unit']}".strip(), round(d["delta"] * 100)]
            for d in out["drivers"]
        ]
        label = {p["display"]: p["p"] for p in out["probabilities"]}
        return verdict_html(cfg, out), label, table

    for s in sliders:
        s.release(run, inputs=sliders, outputs=[card, probs, drivers])
    for btn, values in example_buttons:
        btn.click(lambda v=values: [v[n] for n in names], outputs=sliders).then(
            run, inputs=sliders, outputs=[card, probs, drivers]
        )
    typical.click(lambda: [f["default"] for f in cfg["features"]], outputs=sliders).then(
        run, inputs=sliders, outputs=[card, probs, drivers]
    )
    return sliders, [card, probs, drivers], run


def build():
    warm_up()
    datasets = get_datasets()
    style = dict(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=CSS)
    # Newer Gradio versions take theme/css in launch(); older ones take them in Blocks().
    in_blocks = "css" in inspect.signature(gr.Blocks.__init__).parameters
    demo = gr.Blocks(title="Two Pulses", **(style if in_blocks else {}))
    demo._launch_style = {} if in_blocks else style
    with demo:
        gr.HTML(
            "<div id='hero'><h1>One heartbeat. One pulsar. One pipeline.</h1>"
            "<p>Maternal health risk and pulsar candidate classification, powered by two scikit-learn models.</p></div>"
        )
        for key, cfg in datasets.items():
            with gr.Tab(cfg["title"]):
                sliders, outputs, run = build_tab(key, cfg)
                demo.load(run, inputs=sliders, outputs=outputs)
    return demo


if __name__ == "__main__":
    app = build()
    app.launch(**app._launch_style)
