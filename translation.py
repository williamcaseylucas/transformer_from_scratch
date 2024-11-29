import torch.nn as nn
import torch.optim as optim
import spacy
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator, Example, Dataset
import pandas as pd
import torch
from tqdm.notebook import tqdm
from torchtext.data.metrics import bleu_score
import numpy as np


# for de: conda install -c conda-forge spacy-model-de_core_news_sm
# python -m spacy download de_core_news_md


spacy_de = spacy.load("de_core_news_md")
# python -m spacy download en_core_web_md
spacy_eng = spacy.load("en_core_web_md")


def tokenize_germ(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


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

vars(train_data[0])

german.build_vocab(train_data.src, max_size=10000, min_freq=2)
english.build_vocab(train_data.tgt, max_size=10000, min_freq=2)


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

        # normally you need last part of arange to match last part of expand
        # by unsqueezing(1), we can end with src_seq_len and start with src_seq_len
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

        out = self.fc_out(out)

        return out


device = torch.device("mps")
load_model = False
save_model = True

# Train hyperparams
num_epochs = 5
learning_rate = 3e-4
batch_size = 200

# Model hyperparams
src_vocab_size = len(german.vocab)
trg_vocab_size = len(english.vocab)
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
    # Load german tokenizer
    spacy_ger = spacy_de

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    outputs = [english.vocab.stoi["<sos>"]]
    model.eval()
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [
        english.vocab.itos[idx] for idx in outputs
    ]  # is german for some reason
    # remove start token
    return translated_sentence[1:]


def translate_sentences(model, sentences, german, english, device):
    sentence_len, N = sentences.shape

    targets = torch.LongTensor(
        np.repeat([[english.vocab.stoi["<sos>"]]], N, axis=1)
    ).to(device)

    outputs = torch.zeros((sentence_len, N), dtype=torch.long).to(device)

    model.eval()
    for i in range(sentence_len):
        with torch.no_grad():
            output = model(sentences, targets)  # 1, 200, 5893

        best_guesses = output.argmax(2).squeeze(0)  # (1, 200)

        outputs[i] = best_guesses

        targets = best_guesses.unsqueeze(0)  # (200) -> (1, 200)

        # if best_guess == english.vocab.stoi["<eos>"]:
        #     break

    outputs_flipped = torch.einsum("ij->ji", outputs)  # (11, 200) -> (200, 11)

    translated_sentence = torch.tensor(
        [[english.vocab.itos[idx] for idx in output] for output in outputs_flipped]
    ).transpose(0, 1)
    print("translated_sentence.shape: ", translated_sentence.shape)
    print(
        "translated_sentence: ",
        [" ".join(sentence) for sentence in translated_sentence],
    )
    # remove start token
    return translated_sentence[1:, :]


def bleu(data, model, german, english, device):
    if load_model:
        print("=> Loading checkpoint")
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    targets = []
    outputs = []

    for _, example in tqdm(enumerate(data), total=len(data)):
        # 1. one way of doing this
        # src = vars(example)["src"]
        # tgt = vars(example)["tgt"]
        # 2. another way of dooing this
        src = example.src
        tgt = example.tgt

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token
        targets.append([tgt])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


# def bleu_iterator(data, model, german, english, device):
#     if load_model:
#         print("=> Loading checkpoint")
#         checkpoint = torch.load(model_name)
#         model.load_state_dict(checkpoint["state_dict"])
#         optimizer.load_state_dict(checkpoint["optimizer"])

#     targets = []
#     outputs = []

#     for _, example in tqdm(enumerate(data), total=len(data)):
#         # 1. one way of doing this
#         # src = vars(example)["src"]
#         # tgt = vars(example)["tgt"]
#         # 2. another way of dooing this
#         src = example.src.to(device)
#         tgt = example.tgt.to(device)

#         print("src.shape: ", src.shape)
#         print("tgt.shape: ", tgt.shape)

#         predictions = translate_sentences(model, src, german, english, device)
#         predictions = predictions[:-1, :]  # remove <eos> token

#         tgt = tgt.transpose(0, 1)
#         tgt = np.array(
#             [[english.vocab.itos[word] for word in sentence] for sentence in tgt]
#         ).transpose(0, 1)
#         # print("tgt.shape: ", tgt.shape)
#         print("tgt: ", tgt)

#         targets.append(tgt)
#         outputs.append(predictions)

#     print("final targets: ", targets)
#     print("final outputs: ", outputs)
#     return bleu_score(outputs, targets)


# score = bleu_iterator(test_iterator, model, german, english, device)
# print(f"BLEU score = {score * 100:.2f}")


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

if load_model:
    print("=> Loading checkpoint")
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]

sentence = "ein pferd geht unter einer brücke neben einem boot."
target_sentence = "A horse walks under a bridge next to a boat."

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        print("=> Saving checkpoint")
        torch.save(checkpoint, model_name)

    translated_sentence = translate_sentence(model, sentence, german, english, device)
    print(f"Translated example sentence: \n {translated_sentence}")

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

score = bleu(test_data, model, german, english, device)
print(f"BLEU score = {score * 100:.2f}")
# BLEU score = 31.03
