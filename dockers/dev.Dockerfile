ARG VARIANT="3.9"
FROM justusschock/medical-dl-utils:dev-${VARIANT}

COPY . /tmp/pip-tmp
RUN  pip install --deps-only -f https://download.pytorch.org/whl/cpu/torch_stable.html /tmp/pip-tmp && rm -rf /tmp/pip-tmp