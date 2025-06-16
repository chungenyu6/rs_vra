#!/bin/bash

# =========================
# Define model, port, and GPU mapping
# [MODEL_NAME]="PORT:GPU"
# =========================
declare -A model_port_gpu=(
  # [llama2]="11434:0"
  # [llama3.2-vision]="11435:0"
  [llava:7b-v1.5-fp16]="11436:5" # 16 GB RAM
  # [granite3.2:8b]="11437:1" # 8 GB RAM
  # [granite3.2:8b-instruct-fp16]="11438:0" # 16 GB RAM
  # [qwen2.5:7b-instruct-fp16]="11438:5" # 16 GB RAM
  [qwq]="11439:7" # 20 GB RAM
  [phi4]="11434:5" # 16 GB RAM
  [gemma3:12b-it-fp16]="11433:4" # 24 GB RAM
  # [mistral-small3.1:24b-instruct-2503-q8_0]="11432:5" # 26 GB RAM
  # geochat is currently occupying gpu 6 with 16 GB RAM
)

# =========================
# Create logs directory if it doesn't exist
# =========================
LOG_DIR="serve/logs"
if [ ! -d "$LOG_DIR" ]; then
  echo "Creating logs directory..."
  mkdir -p $LOG_DIR
fi

# =========================
# Start each model server
# =========================
for model in "${!model_port_gpu[@]}"; do
  # Parse port and GPU assignment
  IFS=':' read -r port gpu <<< "${model_port_gpu[$model]}"
  
  echo "Starting Ollama server for model '$model' on port $port using GPU $gpu..."
  
  # Start Ollama serve in background
  CUDA_VISIBLE_DEVICES=$gpu \
  OLLAMA_HOST=127.0.0.1:$port \
  ollama serve > "$LOG_DIR/${model}_${port}.log" 2>&1 &
  
  sleep 2  # Give some time to start the server

  # Preload the model
  echo "Preloading model '$model' at 127.0.0.1:$port..."
  curl -s http://127.0.0.1:$port/api/generate -d "{
    \"model\": \"$model\",
    \"prompt\": \"\",
    \"keep_alive\": -1
  }" > /dev/null

  echo "Model '$model' is ready at http://127.0.0.1:$port"
done

echo "All servers started."