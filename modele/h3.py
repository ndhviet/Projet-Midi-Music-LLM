import os
import json
import argparse
import torch
import mido
from h1 import MusicGPT

import random

PITCH_POOL = [48, 50, 52, 53, 55, 57, 59,
              60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
VEL_POOL   = [64, 80, 96]
DUR_POOL   = [240, 480, 720, 960]
POS_GRID   = [0, 240, 480, 720, 960, 1200, 1440, 1680]

def make_random_seed(n_notes=4):
    tokens = ["BOS", "BAR"]
    positions = sorted(random.sample(POS_GRID, min(n_notes, len(POS_GRID))))
    for pos in positions:
        vel   = random.choice(VEL_POOL)
        pitch = random.choice(PITCH_POOL)
        dur   = random.choice(DUR_POOL)
        tokens += [
            f"POSITION_{pos}",
            f"VELOCITY_{vel}",
            f"NOTE_ON_{pitch}",
            f"DURATION_{dur}",
        ]
    tokens.append("BAR")
    return tokens

# ═══════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════

CONFIG = {
    "checkpoint_dir": "checkpoints",
    "num_songs":      5,
    "max_tokens":     5000,
    "min_tokens":     1000,
    "temperature":    0.85,
    "top_k":          40,
    "top_p":          0.85,
    "block_size":     256,
    "tempo":          100,
    "output_folder":  "generated_music",
    "max_poly":       4,
}

TPQ          = 480
GRID         = TPQ // 4        # 120 ticks
BAR_LENGTH   = TPQ * 4         # 1920 ticks
MIN_DURATION = 480             # 1/8 note minimum — évite le staccato/giật


# ═══════════════════════════════════════════════════════════
#  SEEDS MUSICALES
# ═══════════════════════════════════════════════════════════

SEEDS = [
    # Seed 0
    [
        "BOS", "BAR",
        "POSITION_0",    "VELOCITY_80",  "NOTE_ON_60", "DURATION_480",
        "POSITION_480",  "VELOCITY_80",  "NOTE_ON_64", "DURATION_480",
        "POSITION_960",  "VELOCITY_80",  "NOTE_ON_67", "DURATION_480",
        "POSITION_1440", "VELOCITY_96",  "NOTE_ON_72", "DURATION_960",
        "BAR",
    ],
    # Seed 1
    [
        "BOS", "BAR",
        "POSITION_0",    "VELOCITY_96",  "NOTE_ON_62", "DURATION_240",
        "POSITION_240",  "VELOCITY_80",  "NOTE_ON_64", "DURATION_240",
        "POSITION_480",  "VELOCITY_80",  "NOTE_ON_67", "DURATION_480",
        "POSITION_960",  "VELOCITY_80",  "NOTE_ON_69", "DURATION_480",
        "POSITION_1440", "VELOCITY_96",  "NOTE_ON_71", "DURATION_480",
        "BAR",
    ],
    # Seed 2
    [
        "BOS", "BAR",
        "POSITION_0",    "VELOCITY_64",  "NOTE_ON_48", "DURATION_960",
        "POSITION_0",    "VELOCITY_64",  "NOTE_ON_52", "DURATION_960",
        "POSITION_0",    "VELOCITY_64",  "NOTE_ON_55", "DURATION_960",
        "POSITION_960",  "VELOCITY_96",  "NOTE_ON_67", "DURATION_480",
        "POSITION_1440", "VELOCITY_96",  "NOTE_ON_69", "DURATION_480",
        "BAR",
    ],
    # Seed 3
    [
        "BOS", "BAR",
        "POSITION_0",    "VELOCITY_80",  "NOTE_ON_76", "DURATION_480",
        "POSITION_480",  "VELOCITY_80",  "NOTE_ON_74", "DURATION_480",
        "POSITION_960",  "VELOCITY_80",  "NOTE_ON_72", "DURATION_240",
        "POSITION_1200", "VELOCITY_80",  "NOTE_ON_71", "DURATION_240",
        "POSITION_1440", "VELOCITY_80",  "NOTE_ON_69", "DURATION_480",
        "BAR",
    ],
    # Seed 4
    [
        "BOS", "BAR",
        "POSITION_0",    "VELOCITY_96",  "NOTE_ON_60", "DURATION_720",
        "POSITION_720",  "VELOCITY_80",  "NOTE_ON_62", "DURATION_240",
        "POSITION_960",  "VELOCITY_96",  "NOTE_ON_64", "DURATION_720",
        "POSITION_1680", "VELOCITY_80",  "NOTE_ON_67", "DURATION_240",
        "BAR",
    ],
]


# ═══════════════════════════════════════════════════════════
#  CHARGEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════

def load_model(checkpoint_dir):
    config_path = os.path.join(checkpoint_dir, "config.json")
    model_path  = os.path.join(checkpoint_dir, "best_model.pt")
    vocab_path  = os.path.join(checkpoint_dir, "vocab.txt")

    with open(config_path) as f:
        cfg = json.load(f)
    with open(vocab_path) as f:
        vocab = [l.strip() for l in f if l.strip()]

    token_to_id = {tok: i for i, tok in enumerate(vocab)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}

    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()          else "cpu")

    dummy_weights = torch.ones(cfg["vocab_size"], device=device)

    model = MusicGPT(
        vocab_size   = cfg["vocab_size"],
        d_model      = cfg["d_model"],
        n_heads      = cfg["n_heads"],
        n_layers     = cfg["n_layers"],
        d_ff         = cfg["d_ff"],
        max_seq_len  = cfg["max_seq_len"],
        dropout      = 0.0,
        loss_weights = dummy_weights,
    ).to(device)

    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if "EOS" in token_to_id:
        model.eos_token_id = token_to_id["EOS"]

    print(f"[Device] {device}")
    print(f"[Modèle] {model.count_params():,} paramètres")
    print(f"[Vocab]  {len(vocab)} tokens")
    return model, token_to_id, id_to_token, cfg, device


# ═══════════════════════════════════════════════════════════
#  GÉNÉRATION AUTOREGRESSIVE
# ═══════════════════════════════════════════════════════════

def generate_tokens(model, token_to_id, id_to_token, seed_tokens,
                    max_tokens, min_tokens, temperature, top_k, top_p,
                    block_size, device):

    generated = [t for t in seed_tokens if t in token_to_id]

    for step in range(max_tokens):
        context = generated[-block_size:]
        x = torch.tensor(
            [[token_to_id[t] for t in context]],
            dtype=torch.long, device=device,
        )

        with torch.no_grad():
            logits, _ = model(x)

        logits = logits[0, -1] / max(temperature, 1e-6)

        # Top-k
        if top_k > 0:
            vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < vals[-1]] = float("-inf")

        # Top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum_probs - torch.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = float("-inf")
            logits.scatter_(0, sorted_idx, sorted_logits)

        probs    = torch.softmax(logits, dim=-1)
        next_id  = torch.multinomial(probs, num_samples=1).item()
        next_tok = id_to_token[next_id]
        generated.append(next_tok)

        if step % 500 == 0 and step > 0:
            n_notes = sum(1 for t in generated if t.startswith("NOTE_ON_"))
            print(f"  {step} tokens générés  ({n_notes} notes)")

        if next_tok == "EOS" and len(generated) >= min_tokens:
            print(f"  EOS atteint à {step} tokens")
            break

    return generated


# ═══════════════════════════════════════════════════════════
#  TOKENS → MIDI
# ═══════════════════════════════════════════════════════════

def tokens_to_midi(tokens, output_path, tempo=100, max_poly=4):
    current_bar      = -1
    current_position = 0
    current_velocity = 80
    events = []

    active_notes = set()

    i = 0
    while i < len(tokens):
        tok = tokens[i]

        if tok == "BAR":
            current_bar += 1
            bar_start = current_bar * BAR_LENGTH
            for p, e in list(active_notes):
                if e <= bar_start:
                   active_notes.discard((p, e))

        elif tok.startswith("POSITION_"):
            try:
                current_position = int(tok.split("_", 1)[1]) % BAR_LENGTH
            except ValueError:
                pass

        elif tok.startswith("VELOCITY_"):
            try:
                current_velocity = max(1, min(127, int(tok.split("_", 1)[1])))
            except ValueError:
                pass

        elif tok.startswith("NOTE_ON_"):
            try:
                pitch = int(tok.split("_", 2)[2])
            except ValueError:
                i += 1
                continue

            if not (21 <= pitch <= 108):
                i += 1
                continue

            duration = MIN_DURATION
            if i + 1 < len(tokens) and tokens[i + 1].startswith("DURATION_"):
                try:
                    raw = int(tokens[i + 1].split("_", 1)[1])
                    duration = max(MIN_DURATION, min(raw, BAR_LENGTH * 4))
                    i += 1
                except ValueError:
                    pass

            start = max(0, current_bar) * BAR_LENGTH + current_position
            end   = start + duration

            active_notes = {(p, e) for p, e in active_notes if e > start}

            for p, e in list(active_notes):
                if p == pitch:
                    events.append((start, 'note_off', pitch, 0))
                    active_notes.discard((p, e))
                    break

            if len(active_notes) >= max_poly:
                i += 1
                continue

            active_notes.add((pitch, end))
            events.append((start, 'note_on',  pitch, current_velocity))
            events.append((end,   'note_off', pitch, 0))

        i += 1

    if not events:
        print("  [!] Aucune note créée")
        return False

    events.sort(key=lambda e: (e[0], e[1] == 'note_on'))

    mid   = mido.MidiFile(ticks_per_beat=TPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage('set_tempo', tempo=int(60_000_000 / tempo), time=0))
    track.append(mido.Message('program_change', program=0, channel=0, time=0))

    prev = 0
    for tick, msg_type, pitch, vel in events:
        delta = tick - prev
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, channel=0, time=delta))
        prev = tick

    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.save(output_path)

    n = sum(1 for e in events if e[1] == 'note_on')
    print(f"  [MIDI] {n} notes → {output_path}")
    return True


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", default=CONFIG["checkpoint_dir"])
    p.add_argument("--num_songs",    type=int,   default=CONFIG["num_songs"])
    p.add_argument("--max_tokens",   type=int,   default=CONFIG["max_tokens"])
    p.add_argument("--min_tokens",   type=int,   default=CONFIG["min_tokens"])
    p.add_argument("--temperature",  type=float, default=CONFIG["temperature"])
    p.add_argument("--top_k",        type=int,   default=CONFIG["top_k"])
    p.add_argument("--top_p",        type=float, default=CONFIG["top_p"])
    p.add_argument("--tempo",        type=int,   default=CONFIG["tempo"])
    p.add_argument("--output_folder", default=CONFIG["output_folder"])
    p.add_argument("--max_poly",     type=int,   default=CONFIG["max_poly"])
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    print("\nChargement du modèle...")
    model, token_to_id, id_to_token, cfg, device = load_model(args.checkpoint_dir)

    # Diagnostic seed validity
    for i, seed in enumerate(SEEDS):
        invalid = [t for t in seed if t not in token_to_id]
        if invalid:
            print(f"[WARN] Seed {i} — {len(invalid)} tokens absents du vocab: {invalid}")
        else:
            print(f"[OK]   Seed {i} — tous les tokens valides")

    saved = 0
    for song_idx in range(args.num_songs):
        print(f"\n{'='*50}")
        print(f"  GÉNÉRATION MORCEAU {song_idx + 1} / {args.num_songs}")
        print(f"{'='*50}")

        
        seed = make_random_seed(n_notes=random.randint(3, 5))

        # Random temperature
        temperature = round(random.uniform(0.75, 1.0), 2)
        top_k       = random.choice([25, 30, 40, 50])

        tokens = generate_tokens(
            model       = model,
            token_to_id = token_to_id,
            id_to_token = id_to_token,
            seed_tokens = seed,
            max_tokens  = args.max_tokens,
            min_tokens  = args.min_tokens,
            temperature = temperature,
            top_k       = top_k,
            top_p       = args.top_p,
            block_size  = CONFIG["block_size"],
            device      = device,
        )

        n_notes    = sum(1 for t in tokens if t.startswith("NOTE_ON_"))
        n_bar      = sum(1 for t in tokens if t == "BAR")
        n_pos      = sum(1 for t in tokens if t.startswith("POSITION_"))
        n_duration = sum(1 for t in tokens if t.startswith("DURATION_"))
        n_velocity = sum(1 for t in tokens if t.startswith("VELOCITY_"))
        print(f"  NOTE_ON  : {n_notes}")
        print(f"  BAR      : {n_bar}")
        print(f"  POSITION : {n_pos}")
        print(f"  VELOCITY : {n_velocity}")
        print(f"  DURATION : {n_duration}")
        print(f"  Aperçu   : {tokens[:40]}")

        out_path = os.path.join(args.output_folder, f"song_{song_idx+1:02d}.mid")
        ok = tokens_to_midi(tokens, out_path, tempo=args.tempo, max_poly=args.max_poly)
        if ok:
            saved += 1

    print(f"\n{'='*50}")
    print(f"  {saved}/{args.num_songs} fichiers sauvegardés dans '{args.output_folder}/'")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()


    