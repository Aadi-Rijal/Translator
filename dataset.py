import torch 
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset,load_from_disk
from pathlib import Path

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)
        self.lang_tokens = {
            "en": torch.tensor([tokenizer_tgt.token_to_id("[EN]")], dtype=torch.int64),
            "np": torch.tensor([tokenizer_tgt.token_to_id("[NE]")], dtype=torch.int64),
        }

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Truncate tokens if they are too long
        enc_input_tokens = enc_input_tokens[:self.seq_len - 2]  # Account for lang_token and eos_token
        dec_input_tokens = dec_input_tokens[:self.seq_len - 1]  # Account for eos_token

        # Select the appropriate language token based on the target language
        lang_token = self.lang_tokens[self.tgt_lang]

        # Add padding to ensure sequences are of fixed length
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # Account for lang_token and eos_token
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1  # Account for eos_token

        # Ensure no negative padding (i.e., sentence too long)
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Construct encoder input with language token and eos
        encoder_input = torch.cat(
            [
                lang_token,  # Target language token
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Construct decoder input with language token
        decoder_input = torch.cat(
            [
                lang_token,  # Target language token
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Construct labels with eos token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Ensure all tensors are of the correct sequence length
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["[UNK]", "[PAD]", "[EOS]", "[EN]", "[NE]"])
        tokenizer.train_from_iterator((item[lang] for item in ds), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Load dataset
    ds_raw = load_dataset(config['datasource'])
    print(ds_raw)

    for i in range(3):
      print(ds_raw['train'][i])

    # Build tokenizers for all unique languages in the language pairs
    unique_langs = {lang for pair in config['language_pairs'] for lang in pair}
    tokenizers = {
        lang: get_or_build_tokenizer(config, ds_raw['train'], lang)
        for lang in unique_langs
    }

    # Create datasets for each language pair
    datasets = []
    for src_lang, tgt_lang in config['language_pairs']:
        src_tgt_dataset = BilingualDataset(
            ds=ds_raw['train'],
            tokenizer_src=tokenizers[src_lang],
            tokenizer_tgt=tokenizers[tgt_lang],
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            seq_len=config['seq_len']
        )
        datasets.append(src_tgt_dataset)

    val_dataset = BilingualDataset(
            ds=ds_raw['validation'],
            tokenizer_src=tokenizers['np'],
            tokenizer_tgt=tokenizers['en'],
            src_lang='np',
            tgt_lang='en',
            seq_len=config['seq_len']
        )
    # Combine datasets for all language pairs
    train_dataset = ConcatDataset(datasets)


    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    return train_dataset, val_loader, tokenizers

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])