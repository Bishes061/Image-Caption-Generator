import os
import json
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import multiprocessing as mp

from models import Encoder, DecoderWithAttention
from datasets import CaptionDataset
from utils import (
    AverageMeter,
    accuracy,
    clip_gradient,
    adjust_learning_rate,
    save_checkpoint,
)

# Ensure safe multiprocessing on macOS
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


# Data parameters
data_folder = "data_output"  # folder with data files saved by create_input_files.py
data_name = "flickr8k_5_cap_per_img_5_min_word_freq"  # base name shared by data files

# Model parameters
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0.5

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Training parameters
start_epoch = 0
epochs = 120
epochs_since_improvement = 0
batch_size = 32
workers = 1  # IMPORTANT: >1 breaks with h5py
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.0
alpha_c = 1.0
best_bleu4 = 0.0
print_freq = 100
fine_tune_encoder = False

checkpoint = None  # path to checkpoint, None if none


def main():
    global \
        best_bleu4, \
        epochs_since_improvement, \
        checkpoint, \
        start_epoch, \
        fine_tune_encoder, \
        data_name, \
        word_map

    # Read word map
    word_map_file = os.path.join(data_folder, "WORDMAP_" + data_name + ".json")
    with open(word_map_file, "r") as j:
        word_map = json.load(j)

    # Initialize or load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=emb_dim,
            decoder_dim=decoder_dim,
            vocab_size=len(word_map),
            dropout=dropout,
        )

        decoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, decoder.parameters()),
            lr=decoder_lr,
        )

        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = (
            torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )
            if fine_tune_encoder
            else None
        )

    else:
        # Loading the checkpoint (map to current device)
        checkpoint_data = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint_data["epoch"] + 1
        epochs_since_improvement = checkpoint_data["epochs_since_improvement"]
        best_bleu4 = checkpoint_data["blue-4"]
        decoder = checkpoint_data["decoder"]
        decoder_optimizer = checkpoint_data["decoder_optimizer"]
        encoder = checkpoint_data["encoder"]
        encoder_optimizer = checkpoint_data["encoder_optimizer"]

        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, encoder.parameters()),
                lr=encoder_lr,
            )

    # Move to GPU/MPS if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Dataloaders
    train_loader = DataLoader(
        CaptionDataset(
            data_folder, data_name, "TRAIN", transform=transforms.Compose([normalize])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),  # only useful for CUDA
    )

    val_loader = DataLoader(
        CaptionDataset(
            data_folder, data_name, "VAL", transform=transforms.Compose([normalize])
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training on {device}")

    for epoch in range(start_epoch, epochs):
        # Early stopping
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # Training
        train(
            train_loader,
            encoder,
            decoder,
            criterion,
            encoder_optimizer,
            decoder_optimizer,
            epoch,
        )

        # Validation
        recent_bleu4 = validate(val_loader, encoder, decoder, criterion)

        # Improvement check
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
        else:
            epochs_since_improvement = 0

        # Save
        save_checkpoint(
            data_name,
            epoch,
            epochs_since_improvement,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            recent_bleu4,
            is_best,
        )


def train(
    train_loader,
    encoder,
    decoder,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    epoch,
):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in tqdm(enumerate(train_loader)):
        data_time.update(time.time() - start)

        imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

        # Forward
        imgs = encoder(imgs)
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
            imgs, caps, caplens
        )
        targets = caps_sorted[:, 1:]

        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        loss = criterion(scores, targets)
        loss += alpha_c * (1.0 - alphas.sum(dim=1) ** 2).mean()

        # Backprop
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print(
                f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})"
            )


def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    references, hypotheses = [], []
    start = time.time()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in tqdm(enumerate(val_loader)):
            imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)
            if encoder is not None:
                imgs = encoder(imgs)

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, caplens
            )
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = criterion(scores, targets)
            loss += alpha_c * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(
                    f"Validation: [{i}/{len(val_loader)}]\t"
                    f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})"
                )

            # References
            allcaps = allcaps[sort_ind]
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = [
                    [w for w in c if w not in {word_map["<start>"], word_map["<pad>"]}]
                    for c in img_caps
                ]
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            preds = [p[: decode_lengths[j]] for j, p in enumerate(preds)]
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        bleu4 = corpus_bleu(references, hypotheses)
        print(
            f"\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4}\n"
        )

    return bleu4


if __name__ == "__main__":
    main()
