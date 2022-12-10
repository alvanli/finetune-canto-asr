# syntax=docker/dockerfile:1
FROM huggingface/transformers-pytorch-gpu as base

RUN python3 -m pip install notebook opencc-python-reimplemented
RUN apt-get install -y software-properties-common && apt-get install git-lfs
RUN python3 -m pip install jiwer
# add-apt-repository -y ppa:jonathonf/ffmpeg-4 && apt update && sudo apt install -y ffmpeg
# RUN python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
# RUN python3 -m pip install tensorflow-addons

WORKDIR /exp

CMD ["bash"]