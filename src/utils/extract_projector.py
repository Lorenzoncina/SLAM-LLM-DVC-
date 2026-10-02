import torch
import sys
import os
from collections import OrderedDict

def extract_projector_weights():
    """
    Extracts the encoder_projector state_dict from a saved model's state_dict.
    """
    if len(sys.argv) != 3:
        print("\nUsage: python src/utils/extract_projector.py <path_to_model.pt> <output_path_for_projector.pt>\n")
        sys.exit(1)

    model_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(model_path):
        print(f"Error: Input file not found at '{model_path}'")
        sys.exit(1)

    print(f"Loading model state_dict from: {model_path}")
    full_state_dict = torch.load(model_path, map_location="cpu")

    # Check for nested state_dicts which are common
    if 'model' in full_state_dict:
        full_state_dict = full_state_dict['model']
    elif 'model_state_dict' in full_state_dict:
        full_state_dict = full_state_dict['model_state_dict']

    # Find and extract keys belonging to the encoder_projector
    projector_state_dict = OrderedDict()
    for key, value in full_state_dict.items():
        if key.startswith("encoder_projector."):
            # Remove the "encoder_projector." prefix to create a clean state_dict
            new_key = key.replace("encoder_projector.", "", 1)
            projector_state_dict[new_key] = value

    if not projector_state_dict:
        print("\n❌ Error: Could not find any keys starting with 'encoder_projector.' in the state_dict.")
        print("Please ensure your model.pt file contains the projector weights.")
        sys.exit(1)

    # Save the clean projector weights
    torch.save(projector_state_dict, output_path)
    print(f"✅ Successfully extracted projector weights to: {output_path}")


if __name__ == "__main__":
    extract_projector_weights()