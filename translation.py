import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Example, Dataset
import pandas as pd
import torch
from tqdm.notebook import tqdm

# for de: conda install -c conda-forge spacy-model-de_core_news_sm
# python -m spacy download de_core_news_md


def tokenize_germ(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


spacy_de = spacy.load("de_core_news_sm")
# python -m spacy download en_core_web_md
spacy_eng = spacy.load("en_core_web_sm")

german = Field(
    tokenize=tokenize_germ, lower=True, init_token="<sos>", eos_token="<eos>"
)
english = Field(
    tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>"
)


# .fr -> French Field
# .en -> English Field
# train_data, val_data, test_data = Multi30k.splits(
#     exts=(".de", ".en"), fields=(german, english)
# )
def get_train_test_val() -> list[Dataset, Dataset, Dataset]:
    splits = {"train": "train.jsonl", "validation": "val.jsonl", "test": "test.jsonl"}
    train_data = pd.read_json(
        "hf://datasets/bentrevett/multi30k/" + splits["train"], lines=True
    )
    test_data = pd.read_json(
        "hf://datasets/bentrevett/multi30k/" + splits["test"], lines=True
    )
    val_data = pd.read_json(
        "hf://datasets/bentrevett/multi30k/" + splits["validation"], lines=True
    )

    train_data.shape, test_data.shape, val_data.shape

    res = []
    for data in [train_data, test_data, val_data]:
        english_sentences_tokenized = data["en"].map(lambda x: tokenize_eng(x)).values
        german_sentences_tokenized = data["de"].map(lambda x: tokenize_eng(x)).values

        examples = []
        for g, e in zip(german_sentences_tokenized, english_sentences_tokenized):
            example = Example.fromlist(
                [g, e], fields=[("src", german), ("tgt", english)]
            )
            examples.append(example)

        dataset_res = Dataset(examples, fields=[("src", german), ("tgt", english)])
        res.append(dataset_res)

    return res


train_data, test_data, val_data = get_train_test_val()

english.build_vocab(train_data.src, max_size=10000, min_freq=2)
german.build_vocab(train_data.tgt, max_size=10000, min_freq=2)


class Transformer(nn.Module):
    def __init__(
        self,
        embedding_size,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        forward_expansion,
        dropout,
        max_length,
        device,
    ):
        super(Transformer, self).__init__()

        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_length, embedding_size)

        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_length, embedding_size)

        self.device = device

        self.transformer = nn.Transformer(
            embedding_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            dropout,
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src = [src len, batch size] -> [batch size, src len]
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # (N, src_len)
        return src_mask.to(self.device)

    def forward(self, src, trg):
        (
            src_seq_len,
            N,
        ) = src.shape
        trg_seq_len, N = trg.shape

        src_positions = (
            torch.arange(0, src_seq_len)
            .unsqueeze(1)
            .expand(src_seq_len, N)
            .to(self.device)
        )

        trg_positions = (
            torch.arange(0, trg_seq_len)
            .unsqueeze(1)
            .expand(
                trg_seq_len,
                N,
            )
            .to(self.device)
        )

        embed_src = self.dropout(
            self.src_word_embedding(src) + self.src_position_embedding(src_positions)
        )

        embed_trg = self.dropout(
            self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)
        )

        src_mask = self.make_src_mask(src)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_len).to(
            self.device
        )

        # key_padding_mask: expects N, src_len
        # mask: expects src_len, src_len
        out = self.transformer(
            embed_src,
            embed_trg,
            src_key_padding_mask=src_mask,
            tgt_mask=trg_mask,
        )

        return out


device = torch.device("mps")
load_model = False
save_model = True

# Train hyperparams
num_epochs = 5
learning_rate = 3e-4
batch_size = 32

# Model hyperparams
src_vocab_size = len(english.vocab)
trg_vocab_size = len(german.vocab)
embedding_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1  # seq2seq is lower, 0.5 for fc
src_pad_idx = english.vocab.stoi["<pad>"]
max_len = 100  # sentence length
forward_expansion = 4

writer = SummaryWriter("runs/loss_plot")
step = 0


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_sizes=(batch_size, batch_size, batch_size),
    sort_within_batch=True,  # sort so padding length is similar for data
    sort_key=lambda x: len(x.src),
)

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)
model_name = "my_checkpoint.pth.ptar"


def translate_sentence(model, sentence, german, english, device, max_length=100):
    tokenized_sentence = tokenize_eng(sentence)
    indexed_sentence = [english.vocab.stoi[token] for token in tokenized_sentence]
    indexed_sentence = torch.tensor(indexed_sentence).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(indexed_sentence, indexed_sentence)

    print("outputs.shape", outputs.shape)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

if load_model:
    print("=> Loading checkpoint")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]

sentence = "ein pferd geht unter einer brucke neben einem boot"


for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        print("=> Saving checkpoint")
        torch.save(checkpoint, model_name)

    """translated_sentence = translate_sentence(model, sentence, german, english, device)
    print(f"Translated example sentence: \n {translated_sentence}")"""

    model.train()
    for batch_idx, batch in tqdm(enumerate(train_iterator), total=len(train_iterator)):
        input_data = batch.src.to(device)
        target_data = batch.tgt.to(device)
        output = model(
            input_data, target_data[:-1]
        )  # we want pred word to be next token after input

        output = output.view(-1, output.shape[-1])
        target = target_data[1:].view(-1)
        optimizer.zero_grad()
        loss = criterion(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1

# score = bleu(test_data, model, german, english, device)
# print(f"BLEU score = {score * 100:.2f}")
