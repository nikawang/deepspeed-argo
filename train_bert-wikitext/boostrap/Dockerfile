# Dockerfile_deepspeed
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Install DeepSpeed and other dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-distutils \
    python3.10-venv \
    python3-pip make openssh-server pdsh g++ hostname python3.10-dev cargo git
RUN  python3.10 -m pip install --upgrade pip

# 设置 Python 3.10 为默认的 python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 设置默认的 python 和 pip 指向 python3 和 pip3
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python3.10 -m pip --version

RUN pip install deepspeed

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r ./requirements.txt

# RUN apt-get update && apt-get install -y openssh-server pdsh g++
RUN mkdir -p /var/run/sshd /run/sshd
# COPY id_rsa.pub /root/.ssh/authorized_keys
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/#ChallengeResponseAuthentication yes/ChallengeResponseAuthentication no/' /etc/ssh/sshd_config

#start sshd

# 添加公钥到 /root/.ssh/authorized_keys
RUN mkdir -p /root/.ssh
COPY my_ssh_key.pub /root/.ssh/authorized_keys

COPY my_ssh_key /root/.ssh/id_rsa

# 设置 SSH 密钥的正确权限
RUN chmod 600 /root/.ssh/id_rsa && chmod 600 /root/.ssh/authorized_keys  && chmod 700 /root/.ssh

COPY .netrc /root/.netrc

RUN git clone https://github.com/NVIDIA/apex.git &&  cd apex && python setup.py install --cuda_ext 

EXPOSE 22