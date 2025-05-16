#!/bin/bash

echo "Stopping all Ollama servers..."

# Find all running 'ollama serve' processes
pids=$(ps aux | grep "[o]llama serve" | awk '{print $2}')

if [ -z "$pids" ]; then
  echo "No Ollama servers are running."
else
  echo "Killing processes: $pids"
  kill $pids
  echo "All Ollama servers stopped."
fi