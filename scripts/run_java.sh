#!/bin/bash
#
# Author: s Bostan
# Created on: Nov, 2025
#
# Run Java application

echo "Building Java application..."

# Build with Maven
mvn clean package

# Run the application
java -jar target/adaptive-multimodal-rag-1.0.0.jar

