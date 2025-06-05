import os
import random
from tqdm import tqdm

# FOR MVSA-single

#PROCESSING FROM MultiSentiNet: A Deep Semantic Network for Multimodal Sentiment Analysis

def load_labels(label_file_path):
    """
    Loads and processes labels from the label file.
    
    Expected label file format:
      image_id<TAB>label1,label2
    For example:
      4   positive, positive
      5   positive, negative
      6   positive, neutral
      7   negative, neutral
      
    Processing rules:
      - If the two labels are "positive" and "negative" (in any order), exclude the sample.
      - If one label is "neutral" and the other is either "positive" or "negative", keep the non-neutral label.
      - If both labels are the same, use that label.
      - If there is only one label, use it.
    """
    labels = {}
    with open(label_file_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue  # skip lines without a label field
            img_id = parts[0].strip()
            # Split on comma and remove extra whitespace, and convert to lower-case.
            raw_labels = [lbl.strip().lower() for lbl in parts[1].split(",")]
            # If only one label is present, use it.
            if len(raw_labels) == 1:
                chosen = raw_labels[0]
            elif len(raw_labels) == 2:
                a, b = raw_labels[0], raw_labels[1]
                # Exclude if one is positive and one is negative.
                if (a == "positive" and b == "negative") or (a == "negative" and b == "positive"):
                    continue  # skip this sample
                # If one is neutral and the other is not, choose the non-neutral.
                elif a == "neutral" and b != "neutral":
                    chosen = b
                elif b == "neutral" and a != "neutral":
                    chosen = a
                else:
                    # Otherwise, if both are the same (or neither is neutral) keep either.
                    chosen = a
            else:
                # If there are more than 2 labels, we can decide to take the first one (or apply a custom rule).
                chosen = raw_labels[0]
            # Construct the image filename (assuming .jpg extension).
            labels[f"{img_id}.jpg"] = chosen
    return labels

def save_data(image_files, labels, train_path, val_path, test_path, train_ratio=0.8, val_ratio=0.1):
    """
    Saves the image filename and label pairs into three files.
    Each valid sample is written as: "filename label\n"
    and assigned to the train, validation, or test file based on random split.
    """
    train_file = open(train_path, "w")
    val_file = open(val_path, "w")
    test_file = open(test_path, "w")
    
    for img_file in tqdm(image_files):
        label = labels.get(img_file, None)
        if label is None:
            continue
        entry = f"{img_file} {label}\n"
        r = random.random()
        if r < train_ratio:
            train_file.write(entry)
        elif r < train_ratio + val_ratio:
            val_file.write(entry)
        else:
            test_file.write(entry)
    
    train_file.close()
    val_file.close()
    test_file.close()

if __name__ == "__main__":
    # Set the paths to your directories and files.
    mvsa_dir = "/mimer/NOBACKUP/groups/naiss2024-22-578/MVSA_Single/data/"
    label_file_path = "/mimer/NOBACKUP/groups/naiss2024-22-578/MVSA_Single/labelResultAll.txt"
    
    # Load labels according to our rules.
    labels = load_labels(label_file_path)
    
    # List all JPEG files in the directory that have an associated label.
    image_files = [f for f in os.listdir(mvsa_dir) if f.endswith('.jpg') and f in labels]
    random.shuffle(image_files)  # Shuffle for random splitting.
    
    # Define output file paths.
    train_path = "my_train_mvsa.txt"
    val_path = "my_val_mvsa.txt"
    test_path = "my_test_mvsa.txt"
    
    # Save data using an 80/10/10 split.
    save_data(image_files, labels, train_path, val_path, test_path, train_ratio=0.8, val_ratio=0.1)
