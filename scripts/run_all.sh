#!/bin/bash
#
# Author: s Bostan
# Created on: Nov, 2025
#
# Run all services

echo "Starting AdaptiveMultimodalRAG services..."

# Start Python API
echo "Starting Python API..."
cd demo_api
python app.py &
API_PID=$!
cd ..

# Start UI (if needed)
# echo "Starting UI..."
# cd ui/web
# npm run dev &
# UI_PID=$!
# cd ../..

echo "Services started!"
echo "API PID: $API_PID"
# echo "UI PID: $UI_PID"

# Wait for user interrupt
trap "kill $API_PID" EXIT
wait

