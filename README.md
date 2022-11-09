# gpt-det-deepspeed

This repository is a "WIP" to enable deepspeed GPT models to run on determined.ai without the extra bells and whistles (tracking, metrics, visualization etc.). 

You can still access logs and visualize all metrics via WanDB.

# Pre-Requisites

* Ensure you are connected to determined.ai master (launched via Application Catalog on CoreWeave Cloud)
* In the configuration: gpt_neox_config/small.yml, please change your vocab_path, data_path and load/save paths. (NOTE: We are using a small GPT model but you are welcome to use any configuration for training).
* You can track logs in the determined.ai Web UI. To get WanDB to work correctly, you have to setup an ENV VAR ("WANDB_API_KEY") or modify the function get_wandb_api_key() in deepy.py to just return your API Token. You will be able to visualize all training metrics directly in WanDB.
* You should configure your hostfile path in ```train_deepspeed_launcher.py``` to your determined.ai mounth path. Our default mount path is ```shared_hostfile = "/mnt/finetune-gpt-neox/hostfile.txt"```

# Instructions to Run

* ```git clone --recurse-submodules https://github.com/coreweave/gpt-det-deepseed.git```
* Run: ```det experiment create core_api.yml .``` from the root of the repository.


