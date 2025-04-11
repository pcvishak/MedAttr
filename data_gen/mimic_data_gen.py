import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging # Added for better logging
import argparse

parser = argparse.ArgumentParser(description="Generate QA pairs from MIMIC dataset using LLM.")
parser.add_argument("--data_dir", type=str, default="MIMIC", help="Path to the MIMIC dataset folder.")
parser.add_argument("--output_dir", type=str, default="MIMIC-QA", help="Path to save the generated QA pairs.")

args = parser.parse_args()
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths for input and output
dataset_folder = args.data_dir  # Changed input folder name
qa_folder = args.output_dir  # Changed output folder name



model_id = "meta-llama/Llama-3.3-70B-Instruct" 

# Quantization config (using 4-bit 
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
logging.info(f"Loading model: {model_id}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto", 
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or tokenizer: {e}")
    exit() 


# --- Core Logic ---

def generate_qa_pairs(case_json):
    """
    Generates 2-3 QA pairs based on the input medical case JSON using the LLM.
    Attempts to parse the LLM output as JSON.
    """
    
    prompt = f"""You are an expert medical data analyst. Based on the following Patient Discharge Diagnosis (PDD) data from the MIMIC dataset (in JSON format), generate exactly 2 or 3 relevant Question-Answer (QA) pairs.

    The questions should inquire about specific pieces of information present in the data (like test results, diagnoses, demographics, medications, procedures, etc.).
    The answers should be concise and directly derived from the provided JSON data.

    The output MUST be a valid JSON list containing dictionaries. Each dictionary must have a "question" key and an "answer" key.
    Do not include any introductory text, explanations, apologies, or markdown formatting (like ```json) before or after the JSON list itself. Just output the pure JSON list.

    Example Output Format:
    [
    {{
        "question": "What was the patient's primary diagnosis?",
        "answer": "Specific diagnosis from the JSON"
    }},
    {{
        "question": "What was the result of the X-ray?",
        "answer": "Findings from the X-ray report in the JSON"
    }}
    ]

    Input Medical Case JSON:
    {json.dumps(case_json, indent=2)}

    Generate the QA pairs now based *only* on the information in the JSON above. Output only the JSON list.

    Output JSON:
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    qa_pairs_list = []  # Default to an empty list

    try:
        with torch.no_grad():
            # Generate output using the model
            output = model.generate(
                **inputs,
                max_new_tokens=400,  
                temperature=0.6,    # Set slightly lower temperature for more factual generation
                top_p=0.9,
                do_sample=True,     # Sample to get some creativity
                pad_token_id=tokenizer.eos_token_id  
            )

        # Decode the output
        output_text = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        logging.debug(f"Raw model output:\n{output_text}")

        # --- Attempt to parse the generated text as JSON ---
        try:
            # Find the start and end of the JSON list
            json_start = output_text.find('[')
            json_end = output_text.rfind(']')

            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_string = output_text[json_start:json_end+1]
                qa_pairs_list = json.loads(json_string)

                # Basic validation of the parsed structure
                if not isinstance(qa_pairs_list, list):
                    logging.warning("Parsed output is not a list. Resetting.")
                    qa_pairs_list = []
                else:
                    validated_list = []
                    for item in qa_pairs_list:
                        if isinstance(item, dict) and "question" in item and "answer" in item:
                            validated_list.append(item)
                        else:
                            logging.warning(f"Invalid item found in list: {item}. Skipping.")
                    qa_pairs_list = validated_list
                    # Limit to max 3 pairs if model generated more
                    qa_pairs_list = qa_pairs_list[:3]

            else:
                logging.warning("Could not find valid JSON list delimiters '[' and ']' in the output.")

        except json.JSONDecodeError as e:
            logging.error(f"Failed to decode JSON from model output: {e}\nAttempted to parse:\n{output_text}")
            # Keep qa_pairs_list empty or add an error entry if desired
            # qa_pairs_list = [{"question": "Error", "answer": f"Failed to parse model output."}]
        except Exception as e:
             logging.error(f"Error during JSON parsing/validation: {e}")


    except Exception as e:
        logging.error(f"Error during model generation: {e}")
        # Keep qa_pairs_list empty or add an error entry

    if not qa_pairs_list:
         logging.warning("No valid QA pairs were generated or parsed.")

    return qa_pairs_list # Return the list of QA dictionaries (or empty list on failure)


def process_json_files():
    """
    Iterates through JSON files in the dataset folder, generates QA pairs,
    and saves them to the QA output folder as JSON files.
    """
    processed_count = 0
    error_count = 0
    skipped_count = 0

    for root, _, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(".json"):
                json_path = os.path.join(root, file)
                # Construct output path with .json extension in the qa_folder
                relative_path_base = os.path.relpath(os.path.splitext(json_path)[0], dataset_folder)
                output_path = os.path.join(qa_folder, relative_path_base + ".json") # Save as JSON

                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                logging.info(f"Processing file: {json_path}")

                try:
                    # Read input JSON
                    with open(json_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Generate QA pairs (returns a list of dicts)
                    qa_pairs = generate_qa_pairs(data)

                    # Save output JSON only if QA pairs were successfully generated
                    if qa_pairs:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(qa_pairs, f, indent=4, ensure_ascii=False) # Save the list directly
                        logging.info(f"Successfully generated {len(qa_pairs)} QA pairs. Saved to: {output_path}")
                        processed_count += 1
                    else:
                        logging.warning(f"No QA pairs generated for {json_path}. Skipping save.")
                        skipped_count += 1

                except json.JSONDecodeError as e:
                    logging.error(f"Input file {json_path} is not valid JSON: {e}")
                    error_count += 1
                except Exception as e:
                    logging.error(f"Failed to process file {json_path}: {e}")
                    error_count += 1

    logging.info(f"QA pair generation completed!")
    logging.info(f"Successfully processed: {processed_count} files")
    logging.info(f"Skipped (no QA generated): {skipped_count} files")
    logging.info(f"Errors encountered: {error_count} files")


if __name__ == "__main__":
    process_json_files()