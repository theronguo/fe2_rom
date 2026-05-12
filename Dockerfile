FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-c"]

COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean --all --yes

WORKDIR /workspace
