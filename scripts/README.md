# Scripts

This folder contains environment/runtime helpers only:

- `scripts/docker/` for container workflows (recommended)
- `scripts/env/` for local setup (optional)
- `scripts/jupyter/` for notebook startup helpers

Model implementations and example notebooks are plugin-owned.

Reference plugin:
- https://github.com/jaca230/pioneerML_base_plugin

Jupyter start scripts write logs to `./.runtime/jupyter` by default.
Override with `JUPYTER_LOG_DIR=/path/to/logs`.
Examples: `./scripts/jupyter/start_lab.sh --port 8890 --host 0.0.0.0`,
`./scripts/jupyter/start_notebook.sh -p 8891`, `./scripts/jupyter/stop_lab.sh --timeout 5`.
