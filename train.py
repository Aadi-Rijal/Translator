import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Subset,DataLoader
from tqdm import tqdm
from pathlib import Path
from model import build_transformer
from dataset import get_ds
from dataset import latest_weights_file_path, get_weights_file_path
from validate import run_validation

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)
    train_dataset,val_loader, tokenizers = get_ds(config)

    # Build model
    vocab_src = max(tokenizers[src].get_vocab_size() for src, _ in config['language_pairs'])
    vocab_tgt = max(tokenizers[tgt].get_vocab_size() for _, tgt in config['language_pairs'])
    model = build_transformer(vocab_src, vocab_tgt, config['seq_len'], config['seq_len'], config['d_model']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizers["en"].token_to_id("[PAD]")).to(device)
    writer = SummaryWriter(config["experiment_name"])

    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    print(model_filename)

    # Initialize training state
    initial_epoch = 1
    start_batch = 0

    # Load model if preload is specified
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch']
        global_step = state['global_step']
        start_batch = state.get('batch_index', 0)
    else:
        print('No model to preload, starting from scratch')

    # start_batch = 0


    for epoch in range(initial_epoch, config['num_epochs'] + 1):

        print("epoch:",epoch)
        print("initial epoch:",initial_epoch)
        print("start batch: ",start_batch)

        if epoch == initial_epoch and start_batch > 0:
            indices = list(range(start_batch * config["batch_size"], len(train_dataset)))
            subset = Subset(train_dataset, indices)  # Create a subset
            train_loader = DataLoader(subset, batch_size=config['batch_size'], shuffle=True)
            print("subset len: " ,len(subset))
        else:
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
            print("data set len: ",len(train_dataset))




        torch.cuda.empty_cache()
        model.train()
        epoch_loss = 0



        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):


            optimizer.zero_grad()

            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)

            loss = loss_fn(proj_output.view(-1, vocab_tgt), label.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            writer.add_scalar("Train Loss", loss.item(), global_step)
            global_step += 1

            if global_step % 10 == 0:
              print("loss : ",loss.item())
            if global_step % 500 == 0:
              with torch.no_grad():
                  run_validation(model, val_loader, tokenizers["np"], tokenizers["en"], config["seq_len"], device, print, global_step, writer)

            # Save checkpoint periodically
            if global_step % 2000 == 0:
                model_path = get_weights_file_path(config, f"{epoch:02d}",f"{global_step}")
                torch.save({
                    'epoch': epoch,
                    'batch_index': batch_idx + start_batch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'global_step': global_step
                }, model_path)
                print(f"Checkpoint saved at epoch {epoch}, batch {batch_idx} , global_step: {global_step}",)
                model.eval()
                print(f"--- Loss: {loss.item()} ---")
                with torch.no_grad():
                  run_validation(model, val_loader, tokenizers["np"], tokenizers["en"], config["seq_len"], device, print, global_step, writer)

                model_path = Path(latest_weights_file_path(config))
                model_folder = f"{config['datasource']}_{config['model_folder']}"
                model_folder = Path(model_folder)
                for file in model_folder.glob(f"{config['model_basename']}*.pt"):
                    print(file)
                    print(model_path)
                    if file != model_path:  # Don't delete the newly saved file
                        file.unlink()
                        print("file deleted")

        # Save checkpoint at the end of the epoch
        # start_batch =0

        model_path = get_weights_file_path(config, f"{epoch+1:02d}",f"{global_step}")
        torch.save({
            'epoch': epoch+1,
            'batch_index': 0,  # Reset batch index for the next epoch
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_path)
        print(f"Checkpoint saved at epoch {epoch+1}, batch {batch_idx} , global_step: {global_step}",)
        print(f"Epoch {epoch} completed. Loss: {epoch_loss / len(train_loader)}")

        model_path = Path(latest_weights_file_path(config))
        model_folder = f"{config['datasource']}_{config['model_folder']}"
        model_folder = Path(model_folder)
        for file in model_folder.glob(f"{config['model_basename']}*.pt"):
            print(file)
            print(model_path)
            if file != model_path:  # Don't delete the newly saved file
                file.unlink()
                print("file deleted")

        # Validation step
        model.eval()
        with torch.no_grad():
            run_validation(model, val_loader, tokenizers["np"], tokenizers["en"], config["seq_len"], device, print, global_step, writer)