import os
import tarfile
import random
import shutil

# Define the path to your .tar.gz file
tar_path = '4096_split.tar.gz'

# Paths to the directories where you want to pool all images
base_directories = {
    'train': 'train_images',
    'val': 'val_images',
    'test': 'test_images'
}

# Create the pooled images directories if they don't exist
for directory in base_directories.values():
    os.makedirs(directory, exist_ok=True)

# Extract the tar.gz file and pool images in one go
with tarfile.open(tar_path, 'r:gz') as tar:
    # Get the list of all members (files) in the tar archive
    members = tar.getmembers()
    
    # Separate members by their base directory (train, val, test)
    categorized_members = {base: [] for base in base_directories.keys()}
    for member in members:
        if member.isfile():
            for base in base_directories.keys():
                if member.name.startswith(base + '/'):
                    categorized_members[base].append(member)
                    break

    # Process each category (train, val, test)
    for base, members in categorized_members.items():
        # Find all unique classes (directories) in the current category
        class_names = list(set(os.path.dirname(m.name).split('/')[1] for m in members if m.isfile()))
        
        # Calculate the number of classes to include (95%)
        num_classes_to_include = int(len(class_names) * 0.95)
        
        # Randomly select 95% of the classes
        selected_classes = set(random.sample(class_names, num_classes_to_include))
        
        # Iterate through each member in the current category
        for member in members:
            class_name = os.path.dirname(member.name).split('/')[1]
            if class_name in selected_classes:
                # Extract the image into memory
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    # Create a new name for the image
                    image_name = os.path.basename(member.name)
                    new_image_name = f"{class_name}_{image_name}"
                    new_image_path = os.path.join(base_directories[base], new_image_name)
                    
                    # Write the image directly to the new path
                    with open(new_image_path, 'wb') as out_file:
                        shutil.copyfileobj(extracted_file, out_file)

print("Selected classes have been extracted, copied, and renamed successfully.")
