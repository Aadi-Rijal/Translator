from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "d_model": 512,
        "datasource": 'newt',
        # Define language pairs for bidirectional training
        "language_pairs": [("en", "np"), ("np", "en")],
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",  # {0} will be replaced by language code
        "experiment_name": "runs/tmodel_bidirectional"
    }
