"""Gradio interface - one tab per model (Fish Weight, Body Fat).

Author: Ahmed Shahzad
Developer: Ahmed Shahzad

Run:  python gradio_app.py      then open http://127.0.0.1:7860
"""
import gradio as gr

from common import get_datasets, predict


def build_tab(key: str, cfg: dict):
    gr.Markdown(f"{cfg['description']}  \n*Data: {cfg['source']}*")

    inputs = []
    with gr.Row():
        with gr.Column():
            for f in cfg["features"]:
                if f["kind"] == "category":
                    comp = gr.Dropdown(choices=f["choices"], value=f["default"], label=f["label"])
                else:
                    comp = gr.Number(
                        value=f["default"],
                        label=f"{f['label']}  (training range {f['min']:g} to {f['max']:g})",
                    )
                inputs.append(comp)
            button = gr.Button("Predict", variant="primary")

        with gr.Column():
            result = gr.Markdown(f"### {cfg['target_label']}\n\n*Fill in the form and press Predict.*")
            m = cfg["metrics"]
            gr.Markdown(
                f"**Model quality (test set of {m['n_test']})**  \n"
                f"R² = {m['r2']} | MAE = {m['mae']} {cfg['unit']} | RMSE = {m['rmse']} {cfg['unit']}"
            )

    names = [f["name"] for f in cfg["features"]]

    def run(*vals):
        try:
            out = predict(key, dict(zip(names, vals)))
        except ValueError as err:
            raise gr.Error(str(err))
        text = f"### {cfg['target_label']}\n\n# {out['prediction']} {cfg['unit']}"
        if out["warnings"]:
            text += "\n\n" + "\n".join(f"- ⚠️ {w}" for w in out["warnings"])
        return text

    button.click(run, inputs=inputs, outputs=result)


def build():
    datasets = get_datasets()
    with gr.Blocks(title="Linear Regression Demos") as demo:
        gr.Markdown("# Linear Regression - Fish Weight and Body Fat")
        for key, cfg in datasets.items():
            with gr.Tab(cfg["title"]):
                build_tab(key, cfg)
    return demo


if __name__ == "__main__":
    build().launch()
