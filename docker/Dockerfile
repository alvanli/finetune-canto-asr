# syntax=docker/dockerfile:1
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel as base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y software-properties-common && apt-get install -y git-lfs libsndfile1-dev
RUN python3 -m pip install transformers==4.36.2
RUN python3 -m pip install datasets evaluate tensorboard bitsandbytes==0.37.0 accelerate loralib optimum[onnxruntime]
RUN python3 -m pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install peft==0.7.1
RUN python3 -m pip install notebook opencc-python-reimplemented
RUN python3 -m pip install jiwer librosa soundfile scipy
RUN python3 -m bitsandbytes


WORKDIR /exp

CMD ["bash"]