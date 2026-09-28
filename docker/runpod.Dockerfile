# RunPod serverless image: adds handler dependencies to the CLI prepare runtime.
#
# Build: docker build -f docker/runpod.Dockerfile -t mixlab:runpod .
# Deploy: set as container image in RunPod serverless endpoint config.
#
# The handler accepts JSON jobs with setup/post commands and all mixlab flags.
# See docker/README.md for the full job input reference.

# Set APP_IMAGE to your mixlab CLI image.
ARG APP_IMAGE
FROM ${APP_IMAGE}

# The CLI image runs as uid 10001. A RunPod worker is single-tenant and its
# jobs run their own setup commands and write to the network volume, so this
# image stays root.
USER root

# Override entrypoint from CLI image so we can install packages
ENTRYPOINT []

COPY requirements-runpod.txt /opt/mixlab/requirements-runpod.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip curl wget gdb procps \
    && python3 -m pip install --no-cache-dir -c /opt/mixlab/requirements-prepare.txt -r /opt/mixlab/requirements-runpod.txt \
    && rm -rf /var/lib/apt/lists/*

# RunPod handler + data scripts
COPY scripts/ /scripts/
ENV MIXLAB_SCRIPTS=/scripts

CMD ["python3", "-u", "/scripts/handler.py"]
