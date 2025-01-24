import csv

def contains_special_characters(value):
    """
    Checks if a string contains any special characters .
    Spaces are ignored.
    """
    return not value.replace(" ", "").isalnum()

def validate_csv(input_file, output_file):
    """
    Validates ISBNs and User-IDs in a CSV file by checking for special characters.
    Writes the validated data to a new CSV file with an additional validation column.
    """
    with open(input_file, mode="r", encoding="utf-8") as infile, \
         open(output_file, mode="w", encoding="utf-8", newline="") as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["Valid"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        valid_count = 0
        invalid_count = 0

        for row in reader:
            isbn_valid = not contains_special_characters(row["ISBN"])
            user_id_valid = not contains_special_characters(row["User-ID"])
            is_valid = isbn_valid and user_id_valid

            # Add validation status to the row
            row["Valid"] = is_valid
            writer.writerow(row)

            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
        print(f"Valid rows: {valid_count}")
        print(f"Invalid rows: {invalid_count}")

if __name__ == "__main__":
     #adjust directories if needed
    input_csv = "ratings.csv"  
    output_csv = "validated_ratings.csv" 
    validate_csv(input_csv, output_csv)
