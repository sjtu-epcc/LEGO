# llama3.2-8B
python struct_heatmap.py \
  --model_path /state/partition/model/kehanlu_llama-3.2-8B-Instruct \
  --dataset_name /state/partition/model/TIGER-Lab_WebInstructSub \
  --split train \
  --batch_size 10 \
  --batch_num 250

# mistral-7B
python struct_heatmap.py \
  --model_path /state/partition/model/mistralai_Mistral-7B-Instruct-v0.1 \
  --dataset_name /state/partition/model/TIGER-Lab_WebInstructSub \
  --split train \
  --batch_size 10 \
  --batch_num 250