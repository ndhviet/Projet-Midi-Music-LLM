import os
import time
import math
import json
import torch
import torch.optim as optim
from torch.utils.data import (Dataset, DataLoader, SubsetRandomSampler)
from h1 import MusicGPT


CONFIG = {
    "dataset_file": "dataset.txt",
    "vocab_file":   "vocab.txt",

    # Modèle
    "d_model":      192,
    "n_heads":      6,
    "n_layers":     4,
    "d_ff":         768,
    "max_seq_len":  256,
    "dropout":      0.1,

    # Entraînement
    "batch_size":              32,
    "epochs":                  50,
    "lr":                      3e-4,
    "grad_clip":               1.0,
    "warmup_steps":            500,

    "max_batches_per_epoch":   1000,

    "save_every":              5,
    "checkpoint_dir":          "checkpoints",
    "val_split":               0.1,

    # Arrêt automatique
    "target_loss":             1.5,
    "early_stop_patience":     12,

    # Weighted loss
    "w_note_on":    2.0,
    "w_note_off":   2.0,
    "w_duration":   2.0,
    "w_velocity":   1.0,   # neutral — VELOCITY est token valide et fréquent
    "w_position":   2.5,
}


# ─────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────
class MidiTokenDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.data    = token_ids
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y


def load_data(config):
    with open(config["dataset_file"]) as f:
        encoded = [int(l.strip()) for l in f if l.strip()]
    with open(config["vocab_file"]) as f:
        vocab = [l.strip() for l in f if l.strip()]

    vocab_size = len(vocab)
    print(f"Vocabulaire : {vocab_size} tokens")
    print(f"Dataset     : {len(encoded):,} tokens")

    split     = int(len(encoded) * (1 - config["val_split"]))
    train_ids = encoded[:split]
    val_ids   = encoded[split:]

    seq_len  = config["max_seq_len"]
    train_ds = MidiTokenDataset(train_ids, seq_len)
    val_ds   = MidiTokenDataset(val_ids,   seq_len)

    print(f"Train : {len(train_ds):,} séquences")
    print(f"Val   : {len(val_ds):,} séquences")
    return train_ds, val_ds, vocab, vocab_size


# ─────────────────────────────────────────
#  WEIGHTED LOSS
# ─────────────────────────────────────────
def build_loss_weights(vocab, token_to_id, config, device):
    weights = torch.ones(len(vocab), device=device)
    for tok, idx in token_to_id.items():
        if tok.startswith("NOTE_ON_"):
            weights[idx] = config["w_note_on"]
        elif tok.startswith("NOTE_OFF_"):
            weights[idx] = config["w_note_off"]
        elif tok.startswith("DURATION_"):
            weights[idx] = config["w_duration"]
        elif tok.startswith("VELOCITY_"):
            weights[idx] = config["w_velocity"]
        elif tok.startswith("POSITION_"):
            weights[idx] = config["w_position"]
    return weights


# ─────────────────────────────────────────
#  DIAGNOSTIC — distribution des tokens
# ─────────────────────────────────────────
def analyze_dataset(encoded, id_to_token, n_samples=100_000):
    counts = {}
    for tid in encoded[:n_samples]:
        tok = id_to_token.get(tid, "UNK")
        parts = tok.split("_")
        if len(parts) >= 2:
            prefix = parts[0] + "_" + parts[1]
        else:
            prefix = tok
        counts[prefix] = counts.get(prefix, 0) + 1

    total = sum(counts.values())
    print("\n=== Token Distribution (premiers 100k tokens) ===")
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:15]:
        bar = "█" * int(40 * v / total)
        print(f"  {k:<22} {v:>8,}  ({100*v/total:5.1f}%)  {bar}")
    print()


# ─────────────────────────────────────────
#  SCHEDULER cosine + warmup
# ─────────────────────────────────────────
def get_lr(step, config, total_steps):
    warmup = config["warmup_steps"]
    if step < warmup:
        return config["lr"] * step / max(1, warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    return config["lr"] * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_ds, config, device):
    model.eval()
    max_val = min(300, len(val_ds))
    indices = torch.randperm(len(val_ds))[:max_val]
    loader  = DataLoader(
        val_ds, batch_size=config["batch_size"],
        sampler=SubsetRandomSampler(indices), num_workers=0,
    )
    total, n = 0.0, 0
    for x, y in loader:
        x, y    = x.to(device), y.to(device)
        _, loss = model(x, y)
        total  += loss.item()
        n      += 1
    model.train()
    return total / max(1, n)


# ─────────────────────────────────────────
#  DIAGNOSTIC — prob NOTE_ON après chaque epoch
# ─────────────────────────────────────────
@torch.no_grad()
def note_on_prob(model, token_to_id, device):
    model.eval()

    # Group 1: sau VELOCITY → phải là NOTE_ON (~1.0)
    seeds_vel = [
        ["BOS", "BAR", "POSITION_0",    "VELOCITY_64"],
        ["BOS", "BAR", "POSITION_480",  "VELOCITY_64"],
        ["BOS", "BAR", "POSITION_1680", "VELOCITY_64"],
    ]

    # Group 2: sau DURATION → phải là POSITION hoặc BAR (~0.0 NOTE_ON)
    seeds_dur = [
        ["BOS", "BAR", "POSITION_0",   "VELOCITY_64", "NOTE_ON_60", "DURATION_480"],
        ["BOS", "BAR", "POSITION_480", "VELOCITY_64", "NOTE_ON_64", "DURATION_720"],
        ["BOS", "BAR", "POSITION_0",   "VELOCITY_64", "NOTE_ON_67", "DURATION_960"],
    ]

    def avg_probs(seeds):
        notes, poses, vels = [], [], []
        for seed in seeds:
            ids = [token_to_id[t] for t in seed if t in token_to_id]
            if not ids:
                continue
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1], dim=-1)
            notes.append(sum(probs[token_to_id[t]].item() for t in token_to_id if t.startswith("NOTE_ON_")))
            poses.append(sum(probs[token_to_id[t]].item() for t in token_to_id if t.startswith("POSITION_")))
            vels.append(sum(probs[token_to_id[t]].item() for t in token_to_id if t.startswith("VELOCITY_")))
        avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
        return avg(notes), avg(poses), avg(vels)

    r_vel = avg_probs(seeds_vel)
    r_dur = avg_probs(seeds_dur)

    model.train()

    print(f"  [après VELOCITY] NOTE={r_vel[0]:.3f} POS={r_vel[1]:.3f} VEL={r_vel[2]:.3f}  (attendu NOTE~1.0)")
    print(f"  [après DURATION] NOTE={r_dur[0]:.3f} POS={r_dur[1]:.3f} VEL={r_dur[2]:.3f}  (attendu POS~0.8)")

    return r_vel[0], r_vel[1], r_vel[2]

# ─────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────
def train():
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()          else "cpu")
    print(f"Device : {device}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    train_ds, val_ds, vocab, vocab_size = load_data(CONFIG)
    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}

    # Analyser la distribution des tokens
    with open(CONFIG["dataset_file"]) as f:
        encoded_full = [int(l.strip()) for l in f if l.strip()]
    analyze_dataset(encoded_full, id_to_token)

    # Construire les poids de loss
    loss_weights = build_loss_weights(vocab, token_to_id, CONFIG, device)
    note_ids = [i for tok, i in token_to_id.items() if tok.startswith("NOTE_ON_")]
    pos_ids  = [i for tok, i in token_to_id.items() if tok.startswith("POSITION_")]
    print(f"Loss weights — NOTE_ON x{CONFIG['w_note_on']}  "
          f"POSITION x{CONFIG['w_position']}")
    print(f"  {len(note_ids)} NOTE_ON tokens, {len(pos_ids)} POSITION tokens\n")

    model = MusicGPT(
        vocab_size   = vocab_size,
        d_model      = CONFIG["d_model"],
        n_heads      = CONFIG["n_heads"],
        n_layers     = CONFIG["n_layers"],
        d_ff         = CONFIG["d_ff"],
        max_seq_len  = CONFIG["max_seq_len"],
        dropout      = CONFIG["dropout"],
        loss_weights = loss_weights,   # ← weighted loss
    ).to(device)

    print(f"Paramètres : {model.count_params():,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = CONFIG["lr"],
        betas        = (0.9, 0.95),
        weight_decay = 0.1,
    )

    # ── Reprendre depuis best_model.pt si existe ──────────
    best_path    = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
    best_loss    = float("inf")
    start_epoch  = 0
    patience_cnt = 0
    history      = []
    global_step  = 0

    if os.path.exists(best_path):
        print(f"\nCheckpoint trouvé — reprise de l'entraînement...")
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        best_loss   = ckpt.get("train_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0)
        print(f"  epoch={start_epoch}  best_loss={best_loss:.4f}")

        hist_path = os.path.join(CONFIG["checkpoint_dir"], "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
        global_step = start_epoch * CONFIG["max_batches_per_epoch"]

    # Sauvegarder vocab + config
    with open(os.path.join(CONFIG["checkpoint_dir"], "vocab.txt"), "w") as f:
        for tok in vocab:
            f.write(tok + "\n")
    with open(os.path.join(CONFIG["checkpoint_dir"], "config.json"), "w") as f:
        json.dump({**CONFIG, "vocab_size": vocab_size}, f, indent=2)

    total_steps = CONFIG["epochs"] * CONFIG["max_batches_per_epoch"]

    print(f"\n{'='*55}")
    print(f"  ENTRAÎNEMENT — objectif loss < {CONFIG['target_loss']}")
    print(f"  {CONFIG['epochs']} epochs × {CONFIG['max_batches_per_epoch']} batches")
    print(f"{'='*55}\n")

    for epoch in range(start_epoch + 1, CONFIG["epochs"] + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        max_samples    = CONFIG["max_batches_per_epoch"] * CONFIG["batch_size"]
        max_samples    = min(max_samples, len(train_ds))
        random_indices = torch.randperm(len(train_ds))[:max_samples]
        train_sampler  = SubsetRandomSampler(random_indices)
        train_loader   = DataLoader(
            train_ds, batch_size=CONFIG["batch_size"],
            sampler=train_sampler, num_workers=0, pin_memory=False,
        )

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            lr = get_lr(global_step, CONFIG, total_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1

            if (batch_idx + 1) % 200 == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(f"  Epoch {epoch} | batch {batch_idx+1}/{len(train_loader)}"
                      f" | loss={avg:.4f} | lr={lr:.2e}")

        avg_train = epoch_loss / max(1, len(train_loader))
        avg_val   = evaluate(model, val_ds, CONFIG, device)
        p_note, p_pos, p_vel = note_on_prob(model, token_to_id, device)
        dt        = time.time() - epoch_start

        note_status = ("✓ NOTE_ON OK !" if p_note > 0.15 else
                       "△ en progrès"   if p_note > 0.05 else
                       "✗ trop faible")

        print(f"\nEpoch {epoch:3d}/{CONFIG['epochs']} | "
              f"train={avg_train:.4f} | val={avg_val:.4f} | {dt:.0f}s")
        print(f"  NOTE_ON={p_note:.3f}  POSITION={p_pos:.3f}  "
              f"VELOCITY={p_vel:.3f}  {note_status}\n")

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_train, 5),
            "val_loss":   round(avg_val,   5),
            "note_prob":  round(p_note,    4),
            "pos_prob":   round(p_pos,     4),
            "vel_prob":   round(p_vel,     4),
        })

        # ── Sauvegarde meilleur modèle ─────────────────────
        if avg_train < best_loss:
            best_loss    = avg_train
            patience_cnt = 0
            torch.save({
                "model_state": model.state_dict(),
                "config":      CONFIG,
                "vocab_size":  vocab_size,
                "epoch":       epoch,
                "train_loss":  avg_train,
            }, best_path)
            print(f"  ✓ best_model.pt sauvegardé (loss={avg_train:.4f})")
        else:
            patience_cnt += 1

        if epoch % CONFIG["save_every"] == 0:
            torch.save(
                {"model_state": model.state_dict(), "config": CONFIG, "epoch": epoch},
                os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch:03d}.pt"),
            )

        with open(os.path.join(CONFIG["checkpoint_dir"], "history.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ── Arrêts automatiques ────────────────────────────
        if avg_train <= CONFIG["target_loss"]:
            print(f"\n🎯 Objectif atteint : loss={avg_train:.4f} < {CONFIG['target_loss']}")
            print("   Vous pouvez lancer generate.py !")
            break

        if patience_cnt >= CONFIG["early_stop_patience"]:
            print(f"\n⚠ Early stopping : pas d'amélioration depuis {patience_cnt} epochs")
            break

    print(f"\n{'='*55}")
    print(f"  TERMINÉ — meilleur loss : {best_loss:.4f}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    train()