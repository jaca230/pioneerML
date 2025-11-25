#!/usr/bin/env python3
"""
Deep ZenML environment debug tool.

This script checks:
- All ZenML-related environment variables, including the specific ones
  defined inside ZenML source such as ENV_ZENML_REPOSITORY_PATH.
- Global ZenML configuration stored under ~/.config/zenml.
- Repo-level .zen/.zenml.
- ZenML global client state before & after activate_root().
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# 1. System Info
# --------------------------------------------------------------------------------------
print("\n======================")
print("    SYSTEM INFO")
print("======================")
print("CWD:", Path.cwd())
print("PYTHONPATH:", os.environ.get("PYTHONPATH", "<not set>"))

# --------------------------------------------------------------------------------------
# 2. ZenML-related environment variables (explicit + wildcard)
# --------------------------------------------------------------------------------------

ZENML_ENV_NAMES = [
    "ZENML_ACTIVE_PROJECT_ID",
    "ZENML_ACTIVE_STACK_ID",
    "ZENML_ENABLE_REPO_INIT_WARNINGS",
    "ZENML_REPOSITORY_PATH",
    "ZENML_SERVER",
]

print("\n===============================")
print(" EXPLICIT ZENML ENV VAR CHECK")
print("===============================")

found_explicit = False
for name in ZENML_ENV_NAMES:
    if name in os.environ:
        print(f"{name} = {os.environ[name]}")
        found_explicit = True

if not found_explicit:
    print("No explicit ZENML_* environment variables set.")

print("\n===============================")
print(" ALL ZENML_* ENV VARS")
print("===============================")
zenml_envs = {k: v for k, v in os.environ.items() if k.startswith("ZENML")}
if zenml_envs:
    for k, v in zenml_envs.items():
        print(f"{k} = {v}")
else:
    print("No ZENML_* environment variables found.")

# --------------------------------------------------------------------------------------
# 3. Detect expected project root
# --------------------------------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

print("\n===============================")
print(" EXPECTED PROJECT ROOT")
print("===============================")
print("Expected:", project_root)

print("\nChecking for repo directories at expected root:")
print("  .zen exists?   ", (project_root / ".zen").exists())
print("  .zenml exists? ", (project_root / ".zenml").exists())

# --------------------------------------------------------------------------------------
# 4. Global ZenML user config (~/.config/zenml)
#    Also surface the actual config.yaml location and key lines.
# --------------------------------------------------------------------------------------

print("\n===============================")
print(" GLOBAL ZENML CONFIG (~/.config/zenml)")
print("===============================")

try:
    from zenml.config.global_config import GlobalConfiguration
    gc = GlobalConfiguration()

    print("Global config object loaded.")
    print("Global repository_root     :", getattr(gc, "repository_root", None))
    print("Global config directory    :", getattr(gc, "config_directory", None))

    store_cfg = getattr(gc, "store_configuration", None)
    if store_cfg:
        print("Global store type          :", getattr(store_cfg, "type", None))
        print("Global store URL           :", getattr(store_cfg, "url", None))
    else:
        print("No global store configuration found.")

    # Locate and print the config.yaml lines that reference the store.
    config_dir = Path(getattr(gc, "config_directory", Path.home() / ".config" / "zenml"))
    config_file = config_dir / "config.yaml"
    print("\nGlobal config file path    :", config_file)
    if config_file.exists():
        print("config.yaml exists. Relevant lines:")
        for line in config_file.read_text().splitlines():
            if any(key in line for key in ("store", "url", "repository", "project")):
                print("  ", line)
    else:
        print("config.yaml not found at the expected location.")

except Exception as e:
    print("ERROR loading GlobalConfiguration:", e)

# --------------------------------------------------------------------------------------
# 5. ZenML global client BEFORE activate_root()
# --------------------------------------------------------------------------------------

print("\n===============================")
print(" IMPORTING ZenML.Client")
print("===============================")

from zenml.client import Client

before = None
try:
    before = Client()
    print("Client.root BEFORE activate_root():", before.root)
    print("Client.store BEFORE activate_root():", before.zen_store.config.url)
except Exception as e:
    print("ERROR: could not instantiate Client() before activate_root:", e)

# --------------------------------------------------------------------------------------
# 6. activate_root()
# --------------------------------------------------------------------------------------

print("\n===============================")
print(" CALLING activate_root()")
print("===============================")

try:
    Client().activate_root(project_root)
except Exception as e:
    print("ERROR: activate_root() failed:", e)

try:
    after = Client()
    print("Client.root AFTER activate_root():", after.root)
    print("Client.store AFTER activate_root():", after.zen_store.config.url)
except Exception as e:
    print("ERROR: could not instantiate Client() after activate_root:", e)

# --------------------------------------------------------------------------------------
# 7. Scan for stray ZenML directories
# --------------------------------------------------------------------------------------

print("\n===============================")
print(" SCANNING FOR STRAY .zen / .zenml")
print("===============================")

paths = [
    project_root / ".zen",
    project_root / ".zenml",
    project_root / "notebooks/.zen",
    project_root / "notebooks/.zenml",
    project_root / "notebooks/tutorials/.zen",
    project_root / "notebooks/tutorials/.zenml",
]

for p in paths:
    print(f"{str(p):60s} : {'EXISTS' if p.exists() else 'missing'}")

print("\n===============================")
print(" DONE")
print("===============================")
