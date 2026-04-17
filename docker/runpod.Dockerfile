# RunPod serverless image — adds Python + handler on top of the CLI image.
#
# Build: docker build -f docker/runpod.Dockerfile -t mixlab:runpod .
# Deploy: set as container image in RunPod serverless endpoint config.
#
# The handler accepts JSON jobs with setup/post commands and all mixlab flags.
# See docker/README.md for the full job input reference.

# Set APP_IMAGE to your mixlab CLI image.
ARG APP_IMAGE
FROM ${APP_IMAGE}

# Override entrypoint from CLI image so we can install packages
ENTRYPOINT []

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && pip3 install --no-cache-dir tiktoken tokenizers numpy huggingface_hub runpod \
    && rm -rf /var/lib/apt/lists/*

# RunPod handler + data scripts
COPY scripts/ /scripts/
ENV MIXLAB_SCRIPTS=/scripts

CMD ["python3", "-u", "/scripts/handler.py"]
