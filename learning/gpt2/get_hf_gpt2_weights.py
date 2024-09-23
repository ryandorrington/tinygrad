from transformers import GPT2LMHeadModel
import torch

def get_hf_gpt2_weights():
    # Load the pre-trained GPT-2 model
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Get the model's state dictionary
    state_dict = model.state_dict()

    # Save the state dictionary to a .pth file
    torch.save(state_dict, "hf_gpt2_weights.pth")

    print(f"GPT-2 weights saved to hf_gpt2_weights.pth")

if __name__ == "__main__":
    get_hf_gpt2_weights()
