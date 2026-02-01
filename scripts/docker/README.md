# Docker Scripts

## Build

```bash
./scripts/docker/build.sh
```

Optional:

```bash
./scripts/docker/build.sh --tag pioneerml --version 0.1.0
```

## Run

```bash
./scripts/docker/run.sh
```

Static container (persists IDE packages/state):

```bash
./scripts/docker/run.sh --static
```

Forward ports (e.g. Jupyter):

```bash
./scripts/docker/run.sh --static --port 8888:8888
```
