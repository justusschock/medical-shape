ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

RUN pip install \
  coverage>5.2 \
  codecov>=2.1 \
  pytest==6.* \
  pytest-cov>2.10 \
  # pytest-flake8
  pytest-doctestplus>=0.9.0 \
  check-manifest \
  twine>=3.2 \
  mypy>=0.790 \
  phmdoctest>=1.1.1 \
  pre-commit>=1.0

COPY . /tmp/pip-tmp
RUN  cd /tmp/pip-tmp && python /tmp/pip-tmp/setup.py egg_info && \
  pip install -f https://download.pytorch.org/whl/cpu/torch_stable.html -r /tmp/pip-tmp/*.egg-info/requires.txt && \
  cd && \
  rm -rf /tmp/pip-tmp