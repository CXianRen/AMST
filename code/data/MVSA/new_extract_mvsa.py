import os
import shutil
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer

# --------------------------
# Configuration and Paths
# --------------------------

DATASET_PATH=os.environ.get("DATASET_PATH", None)
if DATASET_PATH is None:
    raise ValueError("Please set the DATASET_PATH environment variable to the root directory of your dataset.")


mvsa_dir = DATASET_PATH
data_dir = os.path.join(mvsa_dir, "data")   # Contains the .txt caption files and images

print(f"Using dataset path: {mvsa_dir}")
# Output directories for tokens, padding masks, and images

text_token_dir = os.path.join(mvsa_dir, "text_token", "roberta-base")
visual_target_dir = os.path.join(mvsa_dir, "visual")
os.makedirs(text_token_dir, exist_ok=True)
os.makedirs(visual_target_dir, exist_ok=True)

# Input label files mapping for each split
path_of_this_script = os.path.dirname(os.path.abspath(__file__))

input_base_dir = path_of_this_script
label_files = {
    "train": "my_train_mvsa.txt",
    "val": "my_val_mvsa.txt",
    "test": "my_test_mvsa.txt"
}

# --------------------------
# Initialize the Tokenizer
# --------------------------
# Using BertTokenizer (adjust add_special_tokens as needed)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# You might prefer add_special_tokens=True for proper encoding:
# tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

# --------------------------
# Process Each Split Directly
# --------------------------
for split, filename in label_files.items():
    input_path = os.path.join(input_base_dir, filename)
    
    # Create dedicated subdirectories for token files and visual images
    token_subdir = text_token_dir
    visual_subdir = visual_target_dir
    os.makedirs(token_subdir, exist_ok=True)
    os.makedirs(visual_subdir, exist_ok=True)
    
    # Read the label file, where each line is expected to be: "image_name label"
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    for line in tqdm(lines, desc=f"Processing {split}"):
        line = line.strip()
        if not line:
            continue
        try:
            img_name, image_label = line.split(" ")
        except ValueError:
            print(f"Skipping malformed line: {line}")
            continue

        # Extract an ID from the image filename (e.g., "123" from "123.jpg")
        ID = os.path.splitext(img_name)[0]
        
        # Read the caption from the corresponding text file
        text_file = os.path.join(data_dir, f"{ID}.txt")
        try:
            with open(text_file, "r", encoding="utf-8") as f_text:
                caption = f_text.read().strip()
        except UnicodeDecodeError:
            with open(text_file, "r", encoding="ISO-8859-1") as f_text:
                caption = f_text.read().strip()
        
        # Tokenize the caption
        encoded = tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=256,
            return_tensors="pt",
            add_special_tokens=True  # Change to False if you explicitly don't want special tokens.
        )
        # encoded is a dictionary with keys: input_ids, attention_mask, token_type_ids
        # input_ids: The token indices in the vocabulary
        # attention_mask: Mask to avoid performing attention on padding token indices.
        # token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        # if token_type_ids is 0 for all tokens, it's a single-sentence input.
       
        # Convert numpy arrays to tensors and remove any unwanted extra dimension.
        token_arr = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].float()  # Convert to float if needed
        # Save the tokens and padding mask as .npy files
        token_save_path = os.path.join(token_subdir, f"{ID}_token.pt")
        pm_save_path = os.path.join(token_subdir, f"{ID}_pm.pt")
        torch.save(token_arr, token_save_path)
        torch.save(attention_mask, pm_save_path)
        
        # Copy the corresponding image (assumed to be in data_dir) to the target directory.
        img_source_path = os.path.join(data_dir, img_name)
        img_target_path = os.path.join(visual_subdir, img_name)
        if os.path.exists(img_source_path):
            shutil.copy2(img_source_path, img_target_path)
        else:
            print(f"Warning: Image not found at {img_source_path}")

print("Processing complete!")
