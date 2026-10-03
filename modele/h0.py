import os
import miditoolkit
 
TPQ        = 480
GRID       = TPQ // 4     
BAR_LENGTH = TPQ * 4


DURATION_BINS = [
    240,   # 1/8
    360,   # 1/8 pointée
    480,   # 1/4
    720,   # 1/4 pointée
    960,   # 1/2
    1440,  # 1/2 pointée
    1920,  # 1 (whole)
    2880,  # 1.5
    3840,  # 2 (double whole)
]


VELOCITY_BINS = [16, 32, 48, 64, 80, 96, 112, 127]

def quantize(value, bins):
    """Quantifie la valeur vers le bin le plus proche."""
    return min(bins, key=lambda b: abs(b - value))

def midi_to_tokens(midi_path):
    midi_obj = miditoolkit.MidiFile(midi_path)
    notes = []

    for instrument in midi_obj.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if note.pitch >= 48:
                notes.append(note)

    notes.sort(key=lambda x: x.start)

    tokens = ["BOS"]
    current_bar = -1

    for note in notes:
        # Quantize start de grid
        start    = round(note.start / GRID) * GRID
        duration = round((note.end - note.start) / GRID) * GRID
        duration = max(GRID, duration)

        # Quantize duration et velocity
        dur_q = quantize(duration, DURATION_BINS)
        vel_q = quantize(note.velocity, VELOCITY_BINS)

        bar = start // BAR_LENGTH
        if bar != current_bar:
            tokens.append("BAR")
            current_bar = bar

        position = start % BAR_LENGTH

        tokens.append(f"POSITION_{position}")
        tokens.append(f"VELOCITY_{vel_q}")
        tokens.append(f"NOTE_ON_{note.pitch}")
        tokens.append(f"DURATION_{dur_q}")

    tokens.append("EOS")
    return tokens


def build_dataset(midi_folder):
    midi_files = [
        f for f in os.listdir(midi_folder)
        if f.lower().endswith((".mid", ".midi"))
    ]
 
    if not midi_files:
        print(f"Aucun fichier MIDI trouvé dans '{midi_folder}'")
        return []
 
    all_tokens = []
    for filename in sorted(midi_files):
        path = os.path.join(midi_folder, filename)
        try:
            tokens = midi_to_tokens(path)
            all_tokens.extend(tokens)
            print(f"  ✓ {filename}: {len(tokens)} tokens")
        except Exception as e:
            print(f"  ✗ {filename}: {e}")
 
    return all_tokens


if __name__ == "__main__":
    midi_folder = "mozart_dataset"
 
    print(f"Tokenisation de '{midi_folder}'...")
    all_tokens = build_dataset(midi_folder)
 
    if not all_tokens:
        print("Aucun token généré.")
        exit(1)
 
    vocab          = sorted(set(all_tokens))
    token_to_id    = {tok: idx for idx, tok in enumerate(vocab)}
    encoded        = [token_to_id[t] for t in all_tokens]
 
    with open("vocab.txt", "w") as f:
        for token in vocab:
            f.write(token + "\n")
 
    with open("dataset.txt", "w") as f:
        for x in encoded:
            f.write(str(x) + "\n")
 
    print(f"\n✓ Dataset créé")
    print(f"  Vocabulaire  : {len(vocab)} tokens")
    print(f"  Tokens total : {len(encoded):,}")
 
    # Aperçu du vocabulaire
    print(f"\nAperçu vocabulaire:")
    for tok in vocab[:10]:
        print(f"  {token_to_id[tok]:4d} → {tok}")
    if len(vocab) > 10:
        print(f"  ... ({len(vocab) - 10} autres)")