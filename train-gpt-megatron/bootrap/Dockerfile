FROM nikawang.azurecr.io/deepspeed-boostrap:v3

WORKDIR /app

RUN pip install nltk==3.8.1 &&  pip install torchvision==0.16.2 && pip install wandb

RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    python setup.py install --cuda_ext && cd - 

RUN apt-get install -y iproute2

RUN git clone --branch v6 --single-branch https://github.com/nikawang/Megatron-DeepSpeed.git && \
    cd Megatron-DeepSpeed && \
    python ./setup.py install

