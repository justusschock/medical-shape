ARG  HARDWARE=cuda
FROM justusschock/medical-dl-utils:latest-${HARDWARE}

COPY . /workdir/medical-shape
RUN pip install /workdir/medical-shape
