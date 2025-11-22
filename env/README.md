Environment setups
==================

Pick the workflow that fits your tooling preference. All options install the same dependencies.

Pip (existing)
--------------
- `pip install -r env/requirements.txt`
- Dev/test extras: `pip install -r env/requirements-dev.txt`

Conda
-----
- `conda env create -f env/environment.yml`
- `conda activate pioneerml`

uv (fast venv + resolver)
-------------------------
- Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create and populate a local venv:
  ```bash
  ./env/uv-setup.sh            # installs core deps into .venv
  ./env/uv-setup.sh --dev      # installs dev/test deps too
  source .venv/bin/activate
  ```

Notes
-----
- All methods use the same dependency lists (`env/requirements.txt` / `env/requirements-dev.txt`).
- Keep a single active environment at a time (deactivate conda/venv before switching).
- RTX 50xx GPUs (sm_120) require PyTorch nightly with CUDA 12.8+; the conda env uses pip nightly wheels via `env/environment.yml`. For existing envs you can manually install with:
  ```bash
  pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- If you want a minimal conda + uv flow, use `./env/setup_uv_conda.sh` (or `--dev` for dev/test deps). It creates `pioneerml-uv`, installs uv, then installs dependencies via uv inside that env.
