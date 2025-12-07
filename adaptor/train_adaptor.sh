# llama-factory init
cd LLaMA-Factory
conda create -n adaptor_train python=3.10
pip install -e ".[torch,metrics]" --no-build-isolation

# configure train settings
# modify model_name_or_path in LEGO/adaptor/LLaMA-Factory/examples/train_full/llama3_full_sft.yaml