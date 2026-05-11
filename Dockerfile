FROM condaforge/miniforge3:latest

SHELL ["/bin/bash", "-c"]

COPY environment.yml /tmp/environment.yml
RUN mamba env create -f /tmp/environment.yml && \
    mamba clean --all --yes

ENV PATH=/opt/conda/envs/fenicsx-new/bin:$PATH
ENV CONDA_DEFAULT_ENV=fenicsx-new
ENV CONDA_PREFIX=/opt/conda/envs/fenicsx-new

WORKDIR /workspace
