# start new terminals
# start servers
conda activate adaptor_train
CUDA_VISIBLE_DEVICES=0 MODEL_PATH=/state/partition/model/kehanlu_llama-3.2-8B-Instruct uvicorn ollama_server:app --host 0.0.0.0 --port 11434
CUDA_VISIBLE_DEVICES=1 MODEL_PATH=/state/partition/model/kehanlu_llama-3.2-8B-Instruct uvicorn ollama_server:app --host 0.0.0.0 --port 11436

# run fighting
conda activate colosseum
# 40 times
.\Run-MakeLocal.ps1 -Times 40
