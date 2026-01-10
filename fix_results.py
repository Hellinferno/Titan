import csv

input_file = "submission_final.csv"
output_file = "submission_final_fixed.csv"

print(f"Reading from {input_file}...")

with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.DictReader(infile)
    # Ensure correct fieldnames
    fieldnames = ['Story ID', 'Prediction', 'Rationale']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    
    writer.writeheader()
    
    count = 0
    for row in reader:
        # Check if Rationale contains the separator and Prediction is empty/missing
        prediction = row.get('Prediction', '')
        rationale = row.get('Rationale', '')
        
        if '|||' in rationale:
            parts = rationale.split('|||', 1)
            row['Prediction'] = parts[0].strip()
            row['Rationale'] = parts[1].strip()
        
        writer.writerow(row)
        count += 1

print(f"✅ Processed {count} rows. Saved to {output_file}")
