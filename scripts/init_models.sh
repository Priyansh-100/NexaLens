#!/bin/bash
set -e

echo "🔧 Pulling Ollama models..."

ollama pull llama3.1:8b
ollama pull nomic-embed-text

echo "✅ Models ready"
echo "Available models:"
ollama list