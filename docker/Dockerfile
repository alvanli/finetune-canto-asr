# syntax=docker/dockerfile:1
# FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel as base
FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel as base
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get install -y software-properties-common && apt-get install -y git-lfs libsndfile1-dev
RUN python3 -m pip install transformers
RUN python3 -m pip install datasets evaluate tensorboard bitsandbytes accelerate loralib optimum audiomentations
RUN python3 -m pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install git+https://github.com/huggingface/peft
RUN python3 -m pip install notebook opencc-python-reimplemented
RUN python3 -m pip install jiwer librosa soundfile scipy
RUN python3 -m pip install flash-attn --no-build-isolation
# RUN python3 -m bitsandbytes

RUN git config --global credential.helper software


WORKDIR /exp

CMD ["bash"]