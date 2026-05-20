# **Gomb-Net**
### Revealing atomic identities in bilayer moire materials
### 2024 Oct 22

Gomb-Net finds atoms, layer-wise, in STEM images of moire materials.

Gomb-Net is a multi-branch U-Net which trains using a unique groupwise output handling and combinatorial loss function.

## Setup with uv

This repo uses `pyproject.toml` to describe the Python environment. The synthetic STEM data dependency is now [`pystemsim`](https://github.com/AustinHouston/pystemsim), not `DataGenSTEM`.

`uv` is a Python environment manager. When you run `uv sync`, it reads `pyproject.toml`, creates a local virtual environment named `.venv`, and installs the packages Gomb-Net needs into that environment.

### 1. Install uv

If you do not already have `uv`, install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen your terminal after installing if the `uv` command is not found.

### 2. Create the environment

From the Gomb-Net folder, run:

```bash
uv sync
```

That command creates `.venv/` in this folder. You usually do not edit anything inside `.venv`; it is just where Python and the installed packages live for this project.

### 3. Use the environment

To run a Python command inside the environment:

```bash
uv run python
```

To run a training script:

```bash
uv run python training_scripts/Train_WSSe_model.py
```

To start JupyterLab for the notebooks:

```bash
uv run jupyter lab
```

You can also activate the environment manually:

```bash
source .venv/bin/activate
```

After activation, `python` and `jupyter lab` will use the packages from `.venv`. When you are done, run:

```bash
deactivate
```

If dependencies change later, run `uv sync` again.



Want to use Gomb-Net, but unsure how to start?

Reach out!



Austin Houston

ahoust17@vols.utk.edu



Try running the following notebooks on google colab (button in the notebooks):

Eval_Graphene_model.ipynb

Eval_WSSe_model.ipynb



all experimental data is available through:

https://drive.google.com/file/d/1DyKtrmJ8wNYQg3YEJ8_iXjz6lB_DQfwy/view?usp=sharing
