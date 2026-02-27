import torch
from transformers import GPT2ForSequenceClassification

def inspect_model():
    try:
        model_path = "final_gpt2_5g_classifier"
        print(f"Loading model from {model_path}...")
        model = GPT2ForSequenceClassification.from_pretrained(model_path)
        
        print("\n--- Model Score Layer Inspection ---")
        print(f"Score Layer Type: {type(model.score)}")
        print(f"Score Layer: {model.score}")
        
        if hasattr(model.score, 'bias') and model.score.bias is not None:
            print(f"\nBias shape: {model.score.bias.shape}")
            print(f"Bias values: {model.score.bias}")
            print(f"Bias for Interoperating (Class 2): {model.score.bias[2] if len(model.score.bias) > 2 else 'N/A'}")
        else:
            print("\nNo bias in score layer.")
            
        # Check num labels
        print(f"\nNum Labels: {model.num_labels}")
        print(f"ID2Label: {model.config.id2label}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_model()
