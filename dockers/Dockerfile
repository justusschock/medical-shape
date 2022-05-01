ARG  HARDWARE=cuda
ARG  BRANCH=stable
FROM justusschock/medical-dl-utils:${BRANCH}-${HARDWARE}

COPY . /workdir/medical-shape
RUN pip install /workdir/medical-shape
