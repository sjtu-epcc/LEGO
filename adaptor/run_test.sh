# lm eval
cd lm-evaluation-harness
conda create -n adaptor_eval python=3.11
conda activate adaptor_eval
pip install -e .
pip install "lm_eval[hf]"

# test
# If test original models, do not use --trust_remote_code.
# IF test custom model, use --trust_remote_code.

# mmlu
lm_eval --model hf \
    --model_args pretrained=/state/partition/model/kehanlu_llama-3.2-8B-Instruct \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code

# squadv2
lm_eval --model hf \
    --model_args pretrained=/state/partition/model/kehanlu_llama-3.2-8B-Instruct \
    --tasks squadv2 \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code

# arc_challenge_chat
lm_eval --model hf \
    --model_args pretrained=/state/partition/model/kehanlu_llama-3.2-8B-Instruct \
    --tasks arc_challenge_chat \
    --device cuda:0 \
    --batch_size 8 \
    --trust_remote_code
