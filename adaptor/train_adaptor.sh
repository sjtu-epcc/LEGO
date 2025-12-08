# llama-factory init
cd LLaMA-Factory
conda create -n adaptor_train python=3.10
conda activate adaptor_train
pip install -e ".[torch,metrics]" --no-build-isolation

# configure train settings
# modify model_name_or_path in LEGO/adaptor/LLaMA-Factory/examples/train_full/llama3_full_sft.yaml
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft.yaml