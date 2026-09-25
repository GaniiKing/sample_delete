# describesnap.py
import os
import json
import base64
import time
from datetime import datetime
import ollama
from cvmemory import add_image_to_memory

SNAPS_FOLDER = "snaps"
METADATA_FILE = os.path.join(SNAPS_FOLDER, "metadata.json")
os.makedirs(SNAPS_FOLDER, exist_ok=True)

# Initialize metadata file if it doesn't exist
if not os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "w") as f:
        json.dump([], f, indent=4)

def describe_with_llama(image_path):
    """Generate description using LLaVA-LLaMA3."""
    try:
        with open(image_path, "rb") as img:
            b64_image = base64.b64encode(img.read()).decode("utf-8")

        prompt = "Describe everything important in this image in 2–3 lines."
        response = ollama.chat(
            model="llava-llama3",
            messages=[{"role": "user", "content": prompt, "images": [b64_image]}]
        )
        return response["message"]["content"]
    except Exception as e:
        return f"Error generating description: {e}"

def load_metadata():
    with open(METADATA_FILE, "r") as f:
        return json.load(f)

def save_metadata(metadata):
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

def rename_snap_file(file_path):
    """Rename snap_… files to snapped_…"""
    folder, file_name = os.path.split(file_path)
    if file_name.startswith("snap_"):
        new_name = "snapped_" + file_name[len("snap_"):]
        new_path = os.path.join(folder, new_name)
        os.rename(file_path, new_path)
        return new_path
    return file_path

def watch_snaps_folder():
    metadata = load_metadata()
    existing_files = {entry["image"] for entry in metadata}

    print(f"Watching folder '{SNAPS_FOLDER}' for new images...")

    while True:
        all_snaps = sorted([
            f for f in os.listdir(SNAPS_FOLDER)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        for snap in all_snaps:
            snap_path = os.path.join(SNAPS_FOLDER, snap)

            # ❌ Skip already renamed images
            if snap.startswith("snapped_"):
                continue

            # ❌ Skip already processed images
            if snap_path in existing_files:
                continue

            print(f"Processing new image: {snap_path}")
            description = describe_with_llama(snap_path)

            # 1️⃣ Rename first (so new_path exists)
            new_path = rename_snap_file(snap_path)
            print(f"📌 Renamed '{snap_path}' to '{new_path}'")

            # 2️⃣ Add to your vector memory DB
            add_image_to_memory(new_path, description)

            # 3️⃣ Save metadata
            entry = {
                "image": new_path,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "description": description
            }
            metadata.append(entry)
            save_metadata(metadata)

            print(f"✅ Metadata saved for {new_path}")
            existing_files.add(new_path)


            # Update in metadata and existing_files
            entry["image"] = new_path
            save_metadata(metadata)

            existing_files.add(new_path)      # add renamed file
            existing_files.add(snap_path)     # also add old path so it never reprocesses

            print(f"📌 Renamed '{snap_path}' to '{new_path}'")

        time.sleep(2)

if __name__ == "__main__":
    watch_snaps_folder()
