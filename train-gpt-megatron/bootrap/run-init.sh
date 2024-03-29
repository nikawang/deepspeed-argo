apt-get install -y vim wget

git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext 
cd -
git clone https://github.com/nikawang/Megatron-DeepSpeed.git
cd Megatron-DeepSpeed
python ./setup.py install

cd dataset
wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz
wget https://huggingface.co/gpt2/resolve/main/vocab.json?download=true -O gpt2-vocab.json
wget https://huggingface.co/gpt2/resolve/main/merges.txt?download=true -O gpt2-merges.txt
xz -d oscar-1GB.jsonl.xz

cd -
python tools/preprocess_data.py \
       --input ../dataset/gpt2/oscar-1GB.jsonl \
       --output-prefix ./dataset/gpt2/my-gpt2 \
       --vocab-file ../dataset/gpt2/gpt2-vocab.json \
       --dataset-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file ../dataset/gpt2/gpt2-merges.txt \
       --append-eod --workers 8

cp my-gpt2* ./dataset
