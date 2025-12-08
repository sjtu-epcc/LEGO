# lm eval
cd lm-evaluation-harness
conda create -n adaptor_eval python=3.10
conda activate adaptor_eval
pip install -e .

# test
lm_eval --model hf \
    --model_args pretrained=/path/to/gguf_folder,gguf_file=model-name.gguf,tokenizer=/path/to/tokenizer \
    --tasks squadv2 \
    --device cuda:0 \
    --batch_size 8