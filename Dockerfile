FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN  apt-get update \
  && apt-get install -y wget\
  && rm -rf /var/lib/apt/lists/*
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -q && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    ~/miniconda3/bin/conda init &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh
ENV LC_ALL=C.UTF-8

RUN /root/miniconda3/bin/conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 ignite -c pytorch  &&\
    /root/miniconda3/bin/conda install -y pandas pytz tqdm scikit-learn wandb transformers  pytorch-lightning tk easydict gensim -c conda-forge
ENV PATH=/root/miniconda3/bin:$PATH
RUN pip install performer-pytorch -y 

RUN wandb login 6ce89b37a3fcfc7af3006497a9e48ac15cfcaf5d

CMD ["/bin/bash"]