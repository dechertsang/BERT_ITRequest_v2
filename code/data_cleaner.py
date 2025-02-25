import pandas as pd
import json
import re
import os

def excel_to_json(excel_path, json_path, replacements_json_path):
    """
    Converts an Excel file to JSON, removing specified characters/words.

    Args:
        excel_path: Path to the Excel file.
        json_path: Path to the output JSON file.
        replacements_json_path: Path to a JSON file containing the replacements.
    """

    try:

        # 1. Read Excel file
        excel_path = r"../file/itrequest.xlsx"
        df = pd.read_excel(excel_path)  # Handles .xls and .xlsx

        # 2. Read replacements from JSON
        with open(replacements_json_path, 'r', encoding='utf-8') as f:  # Handle encoding
            replacements = json.load(f)

        # 3. Apply replacements to all string columns
        for col in df.select_dtypes(include=['object']).columns:  # Iterate through string columns
            df[col] = df[col].astype(str).apply(lambda x: replace_chars(x, replacements))  # Convert to string first

        # 4. Convert DataFrame to list of dictionaries (JSON format)
        data = df.to_dict(orient='records')

        # 5. Write to JSON file
        with open(json_path, 'w', encoding='utf-8') as f: # Handle encoding
            json.dump(data, f, indent=4, ensure_ascii=False) # indent for readability, ensure_ascii to handle non-ascii chars

        print(f"Excel file '{excel_path}' converted to JSON '{json_path}' successfully.")

    except FileNotFoundError:
        print(f"Error: File not found: {excel_path or replacements_json_path}")
    except pd.errors.EmptyDataError:
        print(f"Error: Excel file is empty: {excel_path}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in replacements file: {replacements_json_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def replace_chars(text, replacements):
    """Replaces characters/words in a string."""

    if text is None: # handle null/None values
        return ""

    for char_to_replace in replacements:
        # Using regex for more robust replacement (handles special characters)
        text = re.sub(re.escape(char_to_replace), "", text) # escape special characters in replacements
    return text



# if __name__ == "__main__":

excel_file_path = r"../file/itrequest.xlsx"
json_file_path = r"../file/cls_result.json"
replacements_json_path = r"../file/specify_char.json"
# char_replacements = {"&nbsp;","&nbsp", "<br>", "<br/>", "<", "<er", "<er>","\\t", '"'}
excel_to_json(excel_file_path, json_file_path, replacements_json_path)