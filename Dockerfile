# CUDA runtime image enables GPU access when the container is run with --gpus.
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG PIONEERML_VERSION=dev
ENV PIONEERML_VERSION=${PIONEERML_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

# --- Arrow repo (REQUIRED) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    && curl -fsSL https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-$(lsb_release -cs).deb \
       -o /tmp/apache-arrow-apt.deb \
    && apt-get install -y /tmp/apache-arrow-apt.deb \
    && rm /tmp/apache-arrow-apt.deb

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    pkg-config \
    unzip \
    libcurl4-openssl-dev \
    libarrow-dev \
    libparquet-dev \
    libprotobuf-dev \
    libthrift-dev \
    libre2-dev \
    liblz4-dev \
    libbrotli-dev \
    libsnappy-dev \
    libssl-dev \
    libspdlog-dev \
    libtbb-dev \
    nlohmann-json3-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Provide CMake config stubs to avoid Arrow dependency warnings.
RUN mkdir -p /usr/local/lib/cmake/lz4 /usr/local/lib/cmake/re2 /usr/local/lib/cmake/thrift \
    && printf "add_library(lz4::lz4 SHARED IMPORTED)\nset_target_properties(lz4::lz4 PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/liblz4.so)\n" \
       > /usr/local/lib/cmake/lz4/lz4Config.cmake \
    && printf "add_library(re2::re2 SHARED IMPORTED)\nset_target_properties(re2::re2 PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libre2.so)\n" \
       > /usr/local/lib/cmake/re2/re2Config.cmake \
    && printf "add_library(Thrift::thrift SHARED IMPORTED)\nset_target_properties(Thrift::thrift PROPERTIES IMPORTED_LOCATION /usr/lib/x86_64-linux-gnu/libthrift.so)\n" \
       > /usr/local/lib/cmake/thrift/ThriftConfig.cmake

# Help CMake find NVTX3 and NVRTC for Torch builds (no hardcoded host paths).
RUN if [ -f /usr/local/cuda/lib64/libnvrtc.so.12 ]; then \
      ln -sf /usr/local/cuda/lib64/libnvrtc.so.12 /usr/local/cuda/lib64/libnvrtc.so; \
    fi \
    && if [ -d /usr/local/cuda/include/nvtx3 ]; then \
      mkdir -p /usr/local/lib/cmake/nvtx3; \
      printf "add_library(nvtx3::nvtx3 INTERFACE IMPORTED)\nset_target_properties(nvtx3::nvtx3 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES /usr/local/cuda/include)\n" \
        > /usr/local/lib/cmake/nvtx3/nvtx3Config.cmake; \
    fi

ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py311_24.3.0-0-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniconda.sh

ENV PATH="${CONDA_DIR}/bin:${PATH}"
SHELL ["bash", "-lc"]
ENV CUDA_HOME=/usr/local/cuda

WORKDIR /workspace

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY src src
COPY external external
COPY scripts scripts
COPY notebooks notebooks
COPY tests tests

ENV PIP_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"
ENV PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
ENV UV_PIP_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"
ENV UV_PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
ENV PYTHON_VERSION=3.10

RUN ./scripts/env/setup_uv_conda.sh

RUN conda run -n pioneerml bash -lc "cd external/pioneerml_dataloaders && ./scripts/build.sh"

# Initialize ZenML repository for the workspace.
RUN conda run -n pioneerml bash -lc "zenml init"

ENV CONDA_DEFAULT_ENV=pioneerml
ENV PATH="/opt/conda/envs/pioneerml/bin:${PATH}"
ENTRYPOINT ["bash", "-lc"]
CMD ["bash"]

# Set a consistent prompt with version info.
ENV PS1="\\[\\e[96m\\]pioneerml_v${PIONEERML_VERSION}@\\h\\[\\e[0m\\]:\\[\\e[93m\\]\\w\\[\\e[0m\\]\\$ "
RUN printf 'export PS1="\\[\\e[96m\\]pioneerml_v${PIONEERML_VERSION}@\\h\\[\\e[0m\\]:\\[\\e[93m\\]\\w\\[\\e[0m\\]\\$ "\n' \
    > /etc/profile.d/pioneerml_prompt.sh \
    && printf 'export PS1="\\[\\e[96m\\]pioneerml_v${PIONEERML_VERSION}@\\h\\[\\e[0m\\]:\\[\\e[93m\\]\\w\\[\\e[0m\\]\\$ "\n' \
    >> /root/.bashrc
