import json
import random
import uuid
import os
import torch
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

class ProteinInteractionQA:
    def __init__(self, data_filename):
        self.data_filename = data_filename
        self.data = self.load_data()
    
    def load_data(self):
        """Load data from a JSON file."""
        with open(self.data_filename, 'r') as file:
            data = json.load(file)
        return data

    def generate_question_for_protein(self, entry):
        """Generate a question for a given protein based on its interactions."""
        protein_name = entry["Protein_Name"]
        question = f"What proteins does {protein_name} interact with?"
        answers = [interaction.split(" interacts with ")[1].rsplit(" (", 1)[0] for interaction in entry["Interactions"]]
        return question, answers

    def generate_distractors(self, exclude_protein_name, num_distractors=3):
        """Generate distractors from other proteins."""
        all_interactions = []
        for entries in self.data.values():
            for entry in entries:
                if entry["Protein_Name"] != exclude_protein_name:
                    all_interactions += [interaction.split(" interacts with ")[1].rsplit(" (", 1)[0] for interaction in entry["Interactions"]]
        unique_distractors = list(set(all_interactions))
        distractors = random.sample(unique_distractors, min(len(unique_distractors), num_distractors))
        return distractors

    def save_questions_to_json(self, questions_list, filename="generated_questions.json"):
        """Save questions to a JSON file."""
        with open(filename, 'w') as file:
            json.dump(questions_list, file, indent=4)

    def generate_questions(self):
        """Generate questions and save them to a JSON file."""
        questions_list = []
        for entry_id, entries in self.data.items():
            for entry in entries:
                question, correct_answers = self.generate_question_for_protein(entry)
                distractors = self.generate_distractors(entry["Protein_Name"])
                question_entry = {
                    "id": str(uuid.uuid4()),
                    "question": question,
                    "ideal": ', '.join(correct_answers),
                    "distractors": distractors,
                    "sources": [f"https://www.uniprot.org/uniprot/{entry['UniProtKB_ID']}"]
                }
                questions_list.append(question_entry)
        self.save_questions_to_json(questions_list)

class FunctionQA:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def print_memory_usage(self):
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
        cached = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
        print(f"Allocated memory: {allocated:.2f} GB")
        print(f"Cached memory: {cached:.2f} GB")

    def read_json_file(self, file_path):
        """Reads a JSON file and returns its content."""
        with open(file_path, 'r') as file:
            return json.load(file)

    def generate_question_with_mistral(self, protein_name, function_description):
        print(f"Processing protein: {protein_name}")
        prompt = (f"Generate a biologically accurate multiple-choice question to which there is only one answer by explicitly using the protein name '{protein_name}' based on its function as described here: '{function_description}'. Format the output with the question followed by 'Question:', four short answer options labeled (A, B, C, D), and finally specify the correct answer following 'Answer:'. Ensure the answers are concise and correct.")
        print("Before generating question:")
        self.print_memory_usage()
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            generated_outputs = self.model.generate(inputs["input_ids"], max_new_tokens=250, num_return_sequences=1)
            generated_text = self.tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
            print("After generating question:")
            self.print_memory_usage()
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("CUDA out of memory error caught, clearing cache")
                torch.cuda.empty_cache()
                import time
                time.sleep(5)
                print("Retrying the operation...")
                return self.generate_question_with_mistral(protein_name, function_description)
            else:
                raise e
        finally:
            torch.cuda.empty_cache()
            print("After clearing cache:")
            self.print_memory_usage()
        return generated_text.strip()

    def process_generated_text(self, generated_text):
        parts = re.split(r'\n\s*Question:', generated_text, flags=re.IGNORECASE)
        if len(parts) > 1:
            question_and_options = parts[1].strip()
            correct_answer_match = re.search(r'Answer:\s*([A-D])\)', question_and_options)
            correct_answer_label = correct_answer_match.group(1) if correct_answer_match else None
            options_start = re.search(r'\s*\bA\)', question_and_options)
            if options_start:
                question_text = question_and_options[:options_start.start()].strip()
                options_text = question_and_options[options_start.start():].strip()
                options_text_clean = re.sub(r'\s*Answer:\s*[A-D]\).*', '', options_text, flags=re.IGNORECASE).strip()
                full_question_text = f"{question_text} {options_text_clean}"
                correct_answer = None
                distractors = []
                options_list = re.split(r'\s+(?=[A-D]\))', options_text_clean)
                for option in options_list:
                    option_label = option[:2]
                    option_text = option[3:].strip()
                    if option_label == f"{correct_answer_label})":
                        correct_answer = option_text
                    else:
                        distractors.append(option_text)
                return full_question_text, correct_answer, distractors
        else:
            print("Could not clearly identify the question and answer in the generated text.")
            return None, None, []

    def write_in_chunks(self, data, file_path, chunk_size):
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode) as f:
            if mode == 'w':
                f.write('[')
                print(f"Started writing questions to {file_path}")
            else:
                f.seek(f.tell() - 1, os.SEEK_SET)
                f.write(',')
                print(f"Appending questions to {file_path}")
            first_chunk = True
            for i in range(0, len(data), chunk_size):
                if not first_chunk:
                    f.write(',')
                json.dump(data[i:i + chunk_size], f)
                first_chunk = False
            f.write(']')
            print(f"Finished writing batch of {len(data)} questions to {file_path}")

    def generate_and_collect_questions(self, json_content, output_file_path):
        batch_size = 500
        questions_batch = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_entry = {}
            for pdb_id, entries in json_content.items():
                for entry in entries:
                    future = executor.submit(self.generate_question_with_mistral, entry["Protein_Name"], entry["Function"])
                    future_to_entry[future] = entry
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                generated_text = future.result()
                if generated_text:
                    question, correct_answer, distractors = self.process_generated_text(generated_text)
                    if question and correct_answer:
                        question_data = {
                            "id": str(uuid.uuid4()),
                            "question": question,
                            "ideal": correct_answer,
                            "distractors": distractors,
                            "source": f"https://www.uniprot.org/uniprot/{entry['UniProtKB_ID']}"
                        }
                        questions_batch.append(question_data)
                        if len(questions_batch) >= batch_size:
                            self.write_in_chunks(questions_batch, output_file_path, batch_size)
                            questions_batch = []
            if questions_batch:
                self.write_in_chunks(questions_batch, output_file_path, len(questions_batch))
