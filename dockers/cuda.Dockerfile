ARG  MDLU_BASE_TAG=cuda-latest
FROM justusschock/medical-dl-utils:${MDLU_BASE_TAG}

COPY . /workdir/medical-shape
RUN pip install /workdir/medical-shape