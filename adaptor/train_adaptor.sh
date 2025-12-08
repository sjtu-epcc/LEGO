## llama-factory init
cd LLaMA-Factory
conda create -n adaptor_train python=3.10
conda activate adaptor_train
pip install -e ".[torch,metrics]" --no-build-isolation
pip3 install deepspeed==0.16.4

## Mirror:
# pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple

## modeling code:
# Add modeling_xxx.py code to model directory.
# Modify "self.default_skip_layers = list(range(26, 30))" in modeling code.

## train settings:
# modify LEGO/adaptor/LLaMA-Factory/examples/train_full/llama3_full_sft.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml

# modify LEGO/adaptor/LLaMA-Factory/examples/train_full/mistral_full_sft.yaml
export CUDA_VISIBLE_DEVICES=0,1,2,3
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/mistral_full_sft.yaml