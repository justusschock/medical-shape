ARG  MDLU_BASE_TAG=cpu-latest
FROM justusschock/medical-dl-utils:${MDLU_BASE_TAG}

COPY . /workdir/medical-shape
RUN pip install /workdir/medical-shape