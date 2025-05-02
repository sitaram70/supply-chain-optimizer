import csv
import json


def csv_to_json(csv_file_path, json_file_path):
    data = []

    # Read CSV and convert to list of dictionaries
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            data.append(row)

    # Write JSON
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4)

# Example usage
csv_to_json('demands.csv', 'demands.json')
csv_to_json('processing.csv', 'processing.json')
csv_to_json('transportation.csv', 'transportation.json')
