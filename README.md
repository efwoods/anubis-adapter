# anubis-adapter

adapter-training and inference service for the neural nexus

# This is an adapter training and serving api
The purpose of this repository is to allow for users of the Neural Nexus to train adapters and infer from those adapters from data preprocessed from the Anubis API endpoint.
The repository will demo with smaller models providing the infrastructure to scale as the supply meets demand and the improvement of quality is shown.

This endpoint will compute inference using a [meta-llama/Llama-4-Maverick-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) base with fine-tuned adapters using TRL GRPO algorithms and reward functions matching those that establish the level of quality that are implemented in the [Anubis API](github.com/efwoods/anubis).

Currently, this endpoint computes inference using [meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) due to resource limitations.

This repository will scale to implement the following models (with model size from least to greatest): 
 - [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
 - [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct)
 - [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct)

The implementation is the same, and will scale to serve users concurrent requests for both inference and training of adapters.
The adapters are stored per user_id and assistant_id in an s3 bucket on aws. 
Those adapters are loaded with vllm. 
This endpoint uses vllm to train adapters and serve inference currently using runpod.io.
As user demand increases, this application will scale to meet that demand. 
This is the respository for the scaffold test and demo architecture. 

This will first demo unadapted inference, adapter training, adapter storage, adapter attachment, and multi-assistant and concurrent multi-user adapter inference.

# Resource Requirements

## Memory required for and cost of LoRA adapter training (time, capital)

## LoRA adapter size and cost of storage

- S3 Bucket storage cost:
  - $0.023 per GB for the first 50 TB / month
  - $0.022 per GB for next 450 TB /month
  - $0.021 per GB over 500 TB / month

## Base Model size and cost of storage
 - [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct): 
    - SIZE: 11B * 2B = 22 GB
    - VOLUME COST:
      - runpod.io network volume: 50 GB; $3.50 / month
    - GPU COMPUTE COST:
      - 1XA40 (runpod.io) 
        - (9 max)
        - (48 GB vRam; 240 GB combined vRam): 
        - @ $0.44/GPU; 
        - $0.44 / hour

 - [Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct): 
    - SIZE: 109B * 2B = 218 GB
    - VOLUME COST:
      - runpod.io network volume: 250 GB; $17.50 / month
    - GPU COMPUTE COST:
      - 3xA100 SXM (runpod.io)
        - (8 max)
        - (80 GB vRam; 240 GB combined vRam): 
        - @ $1.49/GPU; 
        - $4.47 / hour
        - Time to load model into memory: 30 to 90 minutes

 - [Llama-4-Maverick-17B-128E-Instruct](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct): 
    - SIZE: 402B * 2B = 804 GB
    - VOLUME COST: 
      - runpod.io volume: 1000 GB; $70.00/month
    - GPU COMPUTE COST:
      - 3xH200 SXM (runpod.io)
        - (8 max)
        - (141 GB vRam; 423 GB combined vRam)
        - @ $4.39 / GPU; 
        - $13.17 / hour

## Number of concurrent users served simultaneously multi-adapter inference

## Number of concurrent users served simultaneously with LoRA adapter training

<!-- RUNPOD: https://console.runpod.io/deploy -->