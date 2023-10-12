FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install pytorch-benchmark==0.3.6

COPY benchmark.py reduce_noise.patch  ./
RUN cd /opt/conda/lib/python3.10/site-packages && patch -p1 < /workspace/reduce_noise.patch

ENTRYPOINT ["python", "benchmark.py"]
