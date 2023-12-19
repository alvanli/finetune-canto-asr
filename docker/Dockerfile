# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel as base
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel as base

RUN apt-get -y update && apt-get install -y software-properties-common && apt-get install -y git-lfs libsndfile1-dev
RUN python3 -m pip install git+https://github.com/huggingface/transformers.git

RUN python3 -m pip install datasets evaluate tensorboard bitsandbytes==0.37.0 accelerate loralib optimum[onnxruntime]
RUN python3 -m pip install torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
RUN python3 -m pip install notebook opencc-python-reimplemented
RUN python3 -m pip install jiwer librosa soundfile
RUN python3 -m pip install -q git+https://github.com/huggingface/peft.git@main 
RUN python3 -m bitsandbytes

WORKDIR /exp

CMD ["bash"]