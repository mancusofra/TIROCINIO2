import os
import glob

# Get the current working directory
current_dir = os.getcwd()

# Find all .csv files in the current directory
csv_files = glob.glob(os.path.join(current_dir, '*.csv'))

# Delete each .csv file
for file in csv_files:
    os.remove(file)
    print(f"Deleted {file}")

print("All .csv files have been deleted.")