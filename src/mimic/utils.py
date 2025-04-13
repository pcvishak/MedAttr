import os
import re

import json
from typing import List, Dict, Any


def extract_passages(json_data: Dict[str, Any]) -> List[str]:
    """
    Extracts meaningful, self-contained passages from EHR/PDD data with:
    - Complete clinical context in each passage
    - Merged related information
    - Normalized text formatting
    - No redundant hierarchy
    """
    passages = []
    seen = set()

    def process_text(text: str) -> List[str]:
        """Split text into meaningful chunks while preserving clinical context"""
        # Split on sentence boundaries and newlines
        chunks = re.split(r'(?<=[.!?])\s+|\n', text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def traverse(node: Dict[str, Any], context: List[str] = None) -> None:
        """Recursively process nested structure to build context-aware passages"""
        current_context = context or []
        
        for key, value in node.items():
            if key.startswith("input"):
                continue  # Skip input metadata

            # Extract clean text from key
            parts = key.split("$", 1)
            text_part = parts[0].strip()
            
            # Create new context stack
            new_context = current_context.copy()
            if text_part:
                new_context.append(text_part)

            # Create passage from accumulated context
            if new_context:
                passage = ": ".join(new_context)
                if passage not in seen:
                    seen.add(passage)
                    passages.append(passage)

            # Process nested values
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    if child_key.startswith("input"):
                        # Handle input values with proper context
                        if isinstance(child_value, str):
                            for chunk in process_text(child_value):
                                full_passage = f"{passage}: {chunk}" if passage else chunk
                                if full_passage not in seen:
                                    seen.add(full_passage)
                                    passages.append(full_passage)
                    else:
                        # Continue context accumulation
                        traverse({child_key: child_value}, new_context)

    # Process main structure
    traverse(json_data)

    # Add standalone input fields with clinical context
    for key, value in json_data.items():
        if key.startswith("input") and isinstance(value, str):
            for chunk in process_text(value):
                if chunk not in seen:
                    seen.add(chunk)
                    passages.append(chunk)

    # Post-process to merge short related chunks
    merged_passages = []
    buffer = ""
    for passage in passages:
        if len(buffer.split()) + len(passage.split()) < 15:  # Merge up to 15 words
            buffer += (" " + passage) if buffer else passage
        else:
            if buffer:
                merged_passages.append(buffer)
            buffer = passage
    if buffer:
        merged_passages.append(buffer)

    return [p for p in merged_passages if 15 < len(p.split()) < 100]


if __name__== "__main__":
    # Example usage
    path = "/nfs/kundeshwar/ashutosh/Attribution/COE_Attribution/data/ehr_conv_data/MIMIC/Acute Coronary Syndrome/NSTEMI/11535902-DS-14.json"
    with open(path, "r") as f:
        example_json = json.load(f)

    passages = extract_passages(example_json)
    for i, passage in enumerate(passages):
        print(f"{i+1}\n\n{passage}\n\n")