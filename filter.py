import os
import shutil


# DOSSIER SOURCE
source_folder = "GrandMidiPiano"


# OUTPUT FOLDERS

mozart_folder = "mozart_dataset"
beethoven_folder = "beethoven_dataset"

os.makedirs(mozart_folder, exist_ok=True)
os.makedirs(beethoven_folder, exist_ok=True)

# COMPTEURS

mozart_count = 0
beethoven_count = 0

# =========================
# FILTRAGE DES FICHIERS
# =========================

for filename in os.listdir(source_folder):

    source_path = os.path.join(source_folder, filename)

    filename_lower = filename.lower()

    
    # MOZART
    
    if "mozart" in filename_lower:

        destination = os.path.join(mozart_folder, filename)

        shutil.copy2(source_path, destination)

        mozart_count += 1

        print("🎹 Mozart:", filename)

   
    # BEETHOVEN
   
    elif "beethoven" in filename_lower:

        destination = os.path.join(beethoven_folder, filename)

        shutil.copy2(source_path, destination)

        beethoven_count += 1

        print("🎼 Beethoven:", filename)



print("\n=========================")
print("DONE")
print("=========================")

print("Mozart files:", mozart_count)
print("Beethoven files:", beethoven_count)
