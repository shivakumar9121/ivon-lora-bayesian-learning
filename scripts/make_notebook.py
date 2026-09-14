"""Build a transparent, self-contained Colab notebook from this repository."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [ROOT / "pyproject.toml", ROOT / "requirements.txt", ROOT / "scripts/training_figures.py"]
    for folder in ["src", "configs", "tests"]:
        paths.extend(p for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in [".py", ".json"])
    contents = {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8") for p in sorted(paths)}
    cells = []
    def md(text):
        cells.append({"cell_type":"markdown","metadata":{},"source":text.splitlines(keepends=True)})
    def code(text):
        cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":text.splitlines(keepends=True)})
    md("""# Variational Low-Rank Adaptation Using IVON
## Qwen2.5-0.5B reproduction on ARC-Easy
**Team:** Siddhu, Pathlavath Shiva Kumar, Gugulothu Ganesh.

Select **Runtime > Change runtime type > T4 GPU**, then **Run all**.
This notebook contains a readable snapshot of the project source. It works without GitHub authentication.
The first large cell only writes the displayed source files into `/content/ivon_lora_study`.

**Evidence rule:** the notebook computes results from real training. An unexecuted notebook contains no measured results.
The two-step smoke test checks the pipeline; it cannot establish a calibration benefit.

Sources: [paper](https://arxiv.org/abs/2411.04421), [official code](https://github.com/team-approx-bayes/ivon-lora), [Qwen model](https://huggingface.co/Qwen/Qwen2.5-0.5B), [ARC data](https://huggingface.co/datasets/allenai/ai2_arc).
""")
    assert all("'''" not in text for text in contents.values())
    readable = "{\n" + "\n".join(repr(name) + ": r'''" + text + "'''," for name, text in contents.items()) + "\n}"
    code("from pathlib import Path\nimport os, json, subprocess, sys\nROOT = Path('/content/ivon_lora_study')\nROOT.mkdir(parents=True, exist_ok=True)\nFILES = " + readable + "\nfor relative, content in FILES.items():\n    target = ROOT / relative\n    target.parent.mkdir(parents=True, exist_ok=True)\n    target.write_text(content, encoding='utf-8')\nos.chdir(ROOT)\nprint('Project source ready:', len(FILES), 'files')\n")
    md("## 1. Environment and offline correctness checks\nPyTorch comes from Colab. The code uses FP32 on T4 to keep IVON gradient and posterior updates numerically straightforward. The final-token projection avoids allocating full-vocabulary logits.")
    code("""# Stream subprocess output into Colab and retain a text log for the presentation.
import time
Path('execution_logs').mkdir(exist_ok=True)
def run_logged(label, command):
    print('Starting:', label, flush=True)
    started = time.perf_counter()
    with open('execution_logs/' + label + '.log', 'a') as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    print(label, 'elapsed seconds:', round(time.perf_counter()-started, 1), flush=True)
    if code:
        raise RuntimeError(label + ' failed; read the error above or its execution log.')
run_logged('gpu_check', [sys.executable, '-u', '-c',
    "import torch; print('PyTorch:',torch.__version__); assert torch.cuda.is_available(), 'Select T4 GPU, not TPU'; print('GPU:',torch.cuda.get_device_name(0))"])
# Install in a subprocess so already-imported notebook packages do not interfere.
deps = [line for line in Path('requirements.txt').read_text().splitlines()
        if line and not line.startswith(('#', 'torch'))]
run_logged('install', [sys.executable, '-m', 'pip', 'install', '-q', *deps])
run_logged('project_install', [sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'])
run_logged('correctness_tests', [sys.executable, '-u', '-m', 'pytest', '-q'])
""")
    md("## 2. Actual-model smoke run\nThis downloads the public model and ARC data, runs two optimizer steps per method, then checks evaluation and reporting. Its metrics are debugging evidence only.")
    code("run_logged('gpu_smoke', [sys.executable, '-u', '-m', 'ivon_lora.train', '--config', 'configs/smoke.json', '--output', 'runs/smoke'])\n")
    md("""## 3. Main experiment
The quick profile trains both optimizers for 128 steps on the same 512 training questions, for seeds 21, 42, and 87. Validation uses 128 questions and the held-out test uses 256. IVON predictions include the posterior mean and a 10-sample probability average.

Do not change the configuration after inspecting test metrics. Use `configs/standard.json` and a new output directory for the larger extension. A completed run resumes by skipping finished optimizer/seed pairs. An interrupted pair restarts from initialization.

Runtime depends on Colab availability and GPU speed. Keep this tab open. Results in `/content` are temporary until you download the archive below.
""")
    md("The default `registered_t4` profile preserves the exact model and dataset revisions and hyperparameters of the completed T4 experiment. Runtime package versions are recorded with every run; cross-environment bitwise identity is not assumed.")
    code("PROFILE = 'registered_t4'\nrun_logged('three_seed_training', [sys.executable, '-u', '-m', 'ivon_lora.train', '--config', f'configs/{PROFILE}.json', '--output', f'runs/{PROFILE}'])\n")
    md("## 4. Measured results and calibration\nThe table gives mean and **sample standard deviation**, not standard error. ECE in the CSV is a fraction; multiply by 100 to obtain percentage points. Deltas are IVON minus AdamW. Three seeds offer limited evidence about statistical significance.")
    code("""import pandas as pd
from IPython.display import display, Image
summary = Path(f'runs/{PROFILE}/summary')
run_logged('training_figures', [sys.executable, 'scripts/training_figures.py', f'runs/{PROFILE}'])
display(pd.read_csv(summary / 'comparison.csv'))
display(pd.read_csv(summary / 'paired_deltas.csv'))
display(pd.read_csv(summary / 'training_details.csv'))
display(Image(filename=str(summary / 'training_curves.png')))
display(Image(filename=str(summary / 'comparison.png')))
display(Image(filename=str(summary / 'reliability.png')))
""")
    md("## 5. Download the evidence\nThis archive contains predictions, seed metrics, data IDs, model and dataset revision hashes, training logs and small adapter checkpoints. The base model is not included. Save the executed notebook with **File > Download > .ipynb** as well.")
    code("""import shutil
from google.colab import files
shutil.copytree('execution_logs', ROOT / f'runs/{PROFILE}/execution_logs', dirs_exist_ok=True)
archive = shutil.make_archive('/content/ivon_lora_measured_results', 'zip', ROOT / f'runs/{PROFILE}')
files.download(archive)
print('Completed experiment archive:', archive)
print('Also save this executed notebook: File > Download > .ipynb')
""")
    md("""## Interpretation for the presentation
- If ECE and NLL decrease across paired seeds, describe evidence of better calibration **in this experiment**.
- If only accuracy improves, do not claim improved calibration.
- If ECE decreases but NLL worsens, discuss the disagreement and show the bin sensitivity file.
- If the gap is small relative to seed SD, call the result inconclusive.
- Qwen differs from Llama in model family and pretraining as well as size. This study cannot isolate a pure model-size effect.
- These are probabilities normalized over four answer tokens, not unrestricted generation confidence.
- The original paper's 2.8-point accuracy and 4.6-point ECE improvements describe **IVON @ mean**. They are not this project's results.
""")
    for i, cell in enumerate(cells):
        cell["id"] = f"ivon-cell-{i:02d}"
    notebook = {"cells":cells,"metadata":{"accelerator":"GPU","colab":{"name":"IVON_LoRA_Reproduction.ipynb","provenance":[]},"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    dest = ROOT / "notebooks" / "IVON_LoRA_Reproduction.ipynb"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(notebook,indent=1),encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()
