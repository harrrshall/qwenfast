# a clean linux machine with nothing but base tools: proves `sh code/install.sh` builds and runs
# qwen fast code from scratch on linux with the same pinned toolchain as macos.
#   docker build -f code/test/linux.Dockerfile -t qfc-linux .    (from the repo root)
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git python3 unzip xz-utils openssh-client procps build-essential && rm -rf /var/lib/apt/lists/*
# the stock ubuntu user (uid 1000) matches the host key owner, so a mounted ~/.ssh works
USER ubuntu
WORKDIR /home/ubuntu/host_qwen
COPY --chown=ubuntu:ubuntu code/toolchain.env code/install.sh code/
COPY --chown=ubuntu:ubuntu code/scripts code/scripts
COPY --chown=ubuntu:ubuntu code/patches code/patches
COPY --chown=ubuntu:ubuntu code/config code/config
COPY --chown=ubuntu:ubuntu code/bin code/bin
COPY --chown=ubuntu:ubuntu agent/package.json agent/package-lock.json agent/
COPY --chown=ubuntu:ubuntu agent/src agent/src
COPY --chown=ubuntu:ubuntu code/test code/test
ENV PATH=/home/ubuntu/.local/bin:$PATH
