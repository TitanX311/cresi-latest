import os
import re

def find_missing_file_numbers(folder_path):
    # Pattern to match numbers at the end of the filename (excluding extension)
    # e.g., "file_123.txt" -> captures 123
    pattern = re.compile(r'(\d+)\.[^.]+$|(\d+)$')
    
    found_numbers = []

    # Check if directory exists
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    # Scan all files in the folder
    for filename in os.listdir(folder_path):
        # Ensure it's a file, not a subfolder
        if os.path.isfile(os.path.join(folder_path, filename)):
            match = pattern.search(filename)
            if match:
                # Extract the matched number group
                num_str = match.group(1) or match.group(2)
                found_numbers.append(int(num_str))

    if not found_numbers:
        print("No numbered files found in the directory.")
        return

    # Sort the found numbers to find the range
    found_numbers.sort()
    start, end = found_numbers[0], found_numbers[-1]

    # Create a set of all expected numbers in the range
    all_expected = set(range(start, end + 1))
    missing_numbers = sorted(list(all_expected - set(found_numbers)))

    # Output the results
    print(f"Total files found: {len(found_numbers)}")
    print(f"Numbered range detected: {start} to {end}")
    print("---")
    
    if missing_numbers:
        print(f"Missing numbers ({len(missing_numbers)} total):")
        print(missing_numbers)
    else:
        print("No numbers are missing in the sequence!")

# --- Example Usage ---
# Replace 'your_folder_path_here' with the actual path to your directory
folder_path = '/home/titanx/Desktop/finalData/cresi_data/train/8bit/PS-RGB' 
find_missing_file_numbers(folder_path)
