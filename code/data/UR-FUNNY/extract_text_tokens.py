import os
import torch  # Import PyTorch
from transformers import RobertaTokenizer

# Path to the file that contains all sample information
sample_txt_file = "./ur_funny_all_text_label.txt"

# Output directory where the token and padding mask files will be saved
output_dir = "/mimer/NOBACKUP/groups/naiss2024-22-578/UR-FUNNY/text_token/roberta-base"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize the BERT tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Open the sample file and process it line by line
with open(sample_txt_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # Skip empty lines

        # Each line is assumed to be in the format:
        #   Ses05F_impro07_F013, exc, 60.2400, 64.5700 ### I know they have a really, really great program for what I want to do.
        # We split the line on "###" to separate the metadata from the text.
        if "###" not in line:
            continue  # Skip lines that do not contain the separator

        meta, text = line.split("###", 1)
        text = text.strip()  # This is the text to tokenize

        # Extract the sample name from the metadata (it's the first item before the first comma)
        sample_name = meta.split(",")[0].strip()

        # Tokenize the text.
        # Using return_tensors="pt" returns PyTorch tensors directly.
            
        encoded = tokenizer(
            text,
            padding="max_length",  # Pad to the maximum sequence length
            truncation=True,
            max_length=128,  # Max sequence length
            return_tensors="pt"  # Returns PyTorch tensors
        )

        # Extract token ids and attention mask from the encoded output.
        # The tokenizer returns tensors with a batch dimension, so we take the first (and only) element.
        tokens = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0].float()  # Convert to float if needed

        # Create the padding mask.
        # Here we assume that positions with a value 0 in the attention mask are padded,
        # so subtracting the attention mask from 1 gives a mask with 1.0 at padded positions.
        #padding_mask = 1.0 - attention_mask

        # Create file names based on the sample name with a .pt extension.
        token_file = os.path.join(output_dir, f"{sample_name}_token.pt")
        pm_file = os.path.join(output_dir, f"{sample_name}_pm.pt")

        # Save the token IDs and padding mask as .pt files.
        torch.save(tokens, token_file)
        torch.save(attention_mask, pm_file)

        print(f"Processed sample: {sample_name}")
