Environment setups
==================

Pick the workflow that fits your tooling preference. All options install the same dependencies.

Pip (recommended)
-----------------
- `pip install -r requirements.txt`

Conda
-----
- `conda env create -f env/environment.yml`
- `conda activate pioneerml`

uv (fast venv + resolver)
-------------------------
- Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create and populate a local venv:
  ```bash
  ./env/uv-setup.sh
  source .venv/bin/activate
  ```

Notes
-----
- All dependencies (core + dev tools) are in `requirements.txt` at the repo root
- Keep a single active environment at a time (deactivate conda/venv before switching)
- RTX 50xx GPUs (sm_120) require PyTorch nightly with CUDA 12.8+; the conda env uses pip nightly wheels via `env/environment.yml`. For existing envs you can manually install with:
  ```bash
  pip install --upgrade --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- If you want a minimal conda + uv flow, use `./env/setup_uv_conda.sh`. It creates `pioneerml-uv`, installs uv, then installs dependencies via uv inside that env.
