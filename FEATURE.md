# Instructions:

please create a python scalar api that will accept basemodel inference per user_id assistant_id with media jobs mimicking those found under the @scaffold folder for the creation and monitoring of adapter training as endpoints.

I will need current_user: dict = Depends(get_current_user), as parameters to continue to verify the user that is hitting each api endpoint.

## There needs to be the following endpoints:
  - all /message endpoints (resume logic for human in the loop etc.) (no select_avatar nor message_selected_avatar)
  - /train_adapter
  - /adapter_training_status
  - /adapter_training_progress
  - /cancel_adapter_training_job

## There needs to be the following functions in addition to the required functions for the above endpoints
  - load_basemodel
  - save adapter
  - download adapter
  - attach adapter
  - remove adapter

# Specifications:
  - This will need to be an asynchronous endpoint
  - This endpoint will need to be able to handle concurrent requests from different users for inference
  - This endpoint will need to be able to handle concurrent requests from different users for adapter training
  - I will need to monitor costs with respect to time, storage, and capital and report those metrics

## I will need to know, compute, and report the following:
  - Memory required for and cost of LoRA adapter training (time, capital)
  - LoRA adapter size and cost of storage
  - Number of concurrent users served simultaneously multi-adapter inference
  - Number of concurrent users served simultaneously with LoRA adapter training

### These are sample metrics for beginning to calculate costs:

#### LoRA adapter size and cost of storage

- S3 Bucket storage cost:
  - $0.023 per GB for the first 50 TB / month
  - $0.022 per GB for next 450 TB /month
  - $0.021 per GB over 500 TB / month

#### Base Model size and cost of storage
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


##### Notes and Purpose:
- I will need to know the break even point to support model inference and adapter training and use implementing this endpoint to support this feature.
- Maverick 4 is a 402 B parameter model on par with the quality of many proprietary models as of the current date. The objective is to improve the quality of the response by using trained adapters that tune to real-world data that attends to the features that define that data. The GRPO reward functions are the SAME functions that are used to measure quality and define likeness or difference with respect to an unmodified chatgpt response or a set of direct quotes captured from local and/or online media. 
- This implementation guarantees improvement in quality and likeness at the limiting reagents of cost, time, compute resources infrastructure, and data availability. Combined with the Anubis API, the API endpoint alleviates the pain point of training adapters for the general public while providing an endpoint to serve the general public with quality that maximizes authenticity where there is a scarcity 
- The lack of supply: (there are no known endpoints to train adapters for and serve inference to Llama-4-Maverick-17B-128E-Instruct nor Llama-4-Scout-17B-16E-Instruct models with those trained adapters). OTHERWISE models rely on prompting alone which exposes a gap with respect to maximizing authenticy of the responses when there are proprietary models that are able to be trained to further increase the authenticity of the responses. 

# Project Structure:
<!-- Note: scaffold is a folder for boilerplate/starter/guidance code only -->
.
├── CLAUDE.md
├── .cursor
├── .env.dev
├── .env.example
├── FEATURE.md
├── .gitignore
├── install.sh
├── _.ipynb
├── LICENSE
├── README.md
├── requirements.txt
├── scaffold
│   ├── media_job.py
│   ├── security
│   │   └── auth.py
│   └── webapp.py
├── settings.json
├── src
│   └── api
│       └── webapp.py
└── .vscode
    └── settings.json

