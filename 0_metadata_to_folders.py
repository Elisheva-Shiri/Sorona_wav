import csv
import os
import shutil

# Read the CSV metadata file
# with open('audio\\esc50.csv', 'r') as f:
with open('audio\\UrbanSound8K.csv', 'r') as f:
    reader = csv.DictReader(f)
    metadata = list(reader)

# Get the list of unique categories
categories = set([row['category'] for row in metadata])

# Create a dictionary to map categories to directory names
category_to_dir = {}
for category in categories:
    category_to_dir[category] = os.path.join('audio', category)

# Loop through the files in the audio_mix directory
for filename in os.listdir('audio\\audio_mix'):
    # Get the category of the file from the metadata
    for row in metadata:
        if row['filename'] == filename:
            category = row['category']
            break

    # Check if the category directory exists
    if not os.path.isdir(category_to_dir.get(category)):
        # If not, create the directory
        os.makedirs(category_to_dir.get(category))

    # Move the file to the category directory
    shutil.move(os.path.join('audio\\audio_mix', filename), os.path.join(category_to_dir.get(category), filename))
