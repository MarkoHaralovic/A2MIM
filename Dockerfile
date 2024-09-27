FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

WORKDIR /

RUN pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install openmim mmcv-full

RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN  apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update -y
RUN apt-get install -y git python3-dev 

ENV CUDA_HOME=/usr/local/cuda-11.1/ 
ENV FORCE_CUDA="1"
ENV  CUDA_VISIBLE_DEVICES=0
ENV MAX_JOBS=1

RUN git clone https://github.com/Westlake-AI/openmixup.git
WORKDIR /openmixup
RUN pip install -v -e .

RUN pip install mmdet

# RUN git clone https://github.com/open-mmlab/mmsegmentation.git
# WORKDIR /mmsegmentation
# RUN pip install -v -e .
RUN pip install mmsegmentation

RUN pip install numpy==1.20.3

WORKDIR /
CMD [ "bash" ]