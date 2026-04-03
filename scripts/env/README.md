Environment setups
==================

Local environment setup (optional). Docker is the recommended default workflow.

Pip
-----------------
- `pip install -r requirements.txt`

Conda **RECOMMENDED**
-----
- `./scripts/env/setup_uv_conda.sh`

uv (fast venv + resolver)
-------------------------
- Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create and populate a local venv:
  ```bash
  ./scripts/env/setup_uv_venv.sh
  source .venv/bin/activate
  ```
- The setup scripts install the repo in editable mode (`uv pip install -e .`) so `import pioneerml`
  works without PYTHONPATH hacks, and register a Jupyter kernel named `pioneerml`.

Notes
-----
- All dependencies (core + dev tools) are in `requirements.txt` at the repo root
- Keep a single active environment at a time (deactivate conda/venv before switching)
- If you want a minimal conda + uv flow, use `./scripts/env/setup_uv_conda.sh`.
- The dependency set is pinned to a reproducible CUDA stack (`torch==2.8.0+cu126`) and includes
  `torch-cluster` (required by PURITY `radius_graph`).

Plugin note
-----
- This repo is the framework layer.
- Model-specific examples and notebook workflows are provided by plugins.
- Reference plugin repo: https://github.com/jaca230/pioneerML_base_plugin
