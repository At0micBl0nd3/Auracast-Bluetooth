import pandas as pd
import os

# Function to parse and process individual CSV files
def process_csv(input_file, packet_start):
    # Read CSV file
    df = pd.read_csv(input_file)

    # Add a new column 'Packet' with unique count starting from packet_start
    df['Packet'] = range(packet_start, packet_start + len(df))

    return df

# Function to iterate through files and concatenate the data
def parse_files(file_prefix, num_files, output_file):
    # Initialize an empty DataFrame to store the parsed data
    parsed_data = pd.DataFrame()

    for i in range(1, num_files + 1):
        # Define file names
        filtered_file = f"{i}_filtered.csv"
        good_file = f"{i}_good.csv"

        # Process filtered file and update packet_start
        filtered_data = process_csv(filtered_file, packet_start=len(parsed_data) + 1)

        # Process good file and update packet_start
        good_data = process_csv(good_file, packet_start=len(parsed_data) + 1)

        # Concatenate data to the parsed_data DataFrame
        parsed_data = pd.concat([parsed_data, filtered_data, good_data], ignore_index=True)

    # Save the parsed data to a new CSV file
    parsed_data.to_csv(output_file, index=False)
    print(f"Parsed data saved to {output_file}")

# Set the number of files and output file name
num_files = 90
output_file = "parsed_data.csv"

# Set the directory where the CSV files are located
# Make sure to adjust the path accordingly
# os.chdir("/path/to/csv/files")

# Parse and concatenate files
parse_files(file_prefix="", num_files=num_files, output_file=output_file)

