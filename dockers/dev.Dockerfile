ARG VARIANT="3.9"
FROM justusschock/medical-dl-utils:dev-${VARIANT}

COPY . /tmp/pip-tmp
RUN  cd /tmp/pip-tmp && python /tmp/pip-tmp/setup.py egg_info && \
  pip install -f https://download.pytorch.org/whl/cpu/torch_stable.html -r /tmp/pip-tmp/*.egg-info/requires.txt && \
  cd && \
  rm -rf /tmp/pip-tmp
