import os
import csv

# Map emotion names to numeric labels
EMOTION_MAP = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

def generate_fer_csv(root_dir, output_csv):
    print(f"Scanning folder: {root_dir}")  # Debug print

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["path", "emotion"])  # Header

        for emotion_name, label_idx in EMOTION_MAP.items():
            subfolder = os.path.join(root_dir, emotion_name)
            
            if not os.path.isdir(subfolder):
                print(f"[WARNING] Skipping missing folder: {subfolder}")  # Debug print
                continue
            
            print(f"Processing: {emotion_name} -> Label {label_idx}")  # Debug print

            for img_name in os.listdir(subfolder):
                img_path = os.path.join(emotion_name, img_name)

                # Ensure the file is an image
                ext = img_name.lower().split('.')[-1]
                if ext in ['jpg', 'jpeg', 'png']:
                    writer.writerow([img_path, label_idx])
                else:
                    print(f"[INFO] Skipping non-image file: {img_name}")

    print(f"CSV file generated: {output_csv}")  # Debug print

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prepare_fer_csv.py <train_or_test>")
        sys.exit(1)

    mode = sys.argv[1].lower().strip()  # 'train' or 'test'
    fer_root = f"data/fer2013/{mode}"
    output_csv = f"data/fer2013/fer_{mode}.csv"

    generate_fer_csv(fer_root, output_csv)
