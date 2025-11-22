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
