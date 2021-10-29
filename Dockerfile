FROM nvcr.io/nvidia/pytorch:20.09-py3
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
            git \
            ssh \
            build-essential \
            locales \
            ca-certificates \
            curl \
            unzip \
            vim \
            wget \
            tmux \
            screen \
            htop \
            less \
            pciutils \
            cron
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && rm -rf /root/.conda \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
# Default to utf-8 encodings in python
# Can verify in container with:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en 
ENV LC_ALL en_US.UTF-8
RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin
RUN apt update
RUN apt install -y python3.7 python3-venv python3.7-venv
RUN pip install --upgrade pip
RUN pip install tf_slim==1.1.0 tensorflow-gpu
RUN conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
RUN conda install tensorboard nltk numpy tqdm
RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge notebook
RUN pip install --upgrade sentence-transformers bert_score fastNLP cytoolz sacrebleu transformers datasets
RUN pip install pytorch_transformers multiprocess pyrouge spacy==3.0.3 absl-py scikit-learn rouge rouge_score gensim matplotlib matching tensorboardX pyopenssl accelerate pycorenlp
RUN python -m spacy download en_core_web_lg
RUN python -m nltk.downloader wordnet punkt
RUN apt-get install -y libxml-parser-perl byobu
RUN pip install --upgrade gsutil
RUN apt-get update
RUN apt-get install -y default-jdk
CMD bash
