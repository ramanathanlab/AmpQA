import argparse
import sys
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging
from utils import ProteinInteractionQA, FunctionQA

# Set logging level to error to reduce console clutter
logging.set_verbosity_error()

# Clear any leftover memory from previous computations
torch.cuda.empty_cache()

# Select a specific GPU; here GPU with ID '5' is chosen
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

# Ensure reproducibility
torch.manual_seed(42)

def main():
    parser = argparse.ArgumentParser(description='Generate QA datasets for proteins.')
    parser.add_argument('-t', '--type', type=str, choices=['i', 'f'], help='Type of QA to generate: "i" for interactions, "f" for functions')
    parser.add_argument('-i', '--input', type=str, required=True, help='Input JSON file containing protein data')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory for saving generated questions')
    
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.type == 'i':
        qa_system = ProteinInteractionQA(args.input)
        qa_system.generate_questions()
        output_path = os.path.join(args.output, 'interaction_questions.json')
        qa_system.save_questions_to_json(qa_system.data, output_path)
        print(f"Protein interaction questions generated and saved to {output_path}.")
    elif args.type == 'f':
        # Load the Mistral model
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"GPU details: {torch.cuda.get_device_name(0)}")

        function_qa = FunctionQA(model, tokenizer, device)
        with open(args.input, 'r') as file:
            json_content = json.load(file)
        output_file_path = os.path.join(args.output, 'function_questions.json')
        function_qa.generate_and_collect_questions(json_content, output_file_path)
        print(f"Protein function questions generated and saved to {output_file_path}.")
    else:
        print("Invalid or missing type argument. Please specify -t i for interactions or -t f for functions.")
        sys.exit(1)

if __name__ == "__main__":
    main()