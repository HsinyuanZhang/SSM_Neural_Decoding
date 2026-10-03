FROM python:3.10-slim-bookworm@sha256:2559be987fd64d61badbdafd303ea58a9ccab36d6c3c08bce219e762177d2eca AS runtime

WORKDIR /
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.1 \
 && python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu \
 && python -m pip install --no-cache-dir -r /tmp/requirements-runtime.txt

FROM runtime AS inference
COPY payload /payload
COPY decode.py /decode.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENV PYTHONPATH=/payload PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
    EVAL_DATA_PATH=/dataset/evaluation_data PREDICTION_PATH=/submission/submission.csv \
    PREDICTION_PATH_LOCAL=/tmp/submission.pkl GT_PATH=/tmp/ground_truth.pkl
ENTRYPOINT ["/entrypoint.sh"]
CMD ["--evaluation", "remote", "--model-path", "/payload", "--split", "m1", "--phase", "test", "--batch-size", "4"]
