import os
import tarfile
import random
import shutil

# Define the path to your .tar.gz file
tar_path = 'rwave-1024.tar.gz'
# Path to the directory where you want to pool all images
train_images_directory = 'train_images'

# Create the train_images directory if it doesn't exist
os.makedirs(train_images_directory, exist_ok=True)

# Extract the tar.gz file and pool images in one go
with tarfile.open(tar_path, 'r:gz') as tar:
    # Get the list of all members (files) in the tar archive
    members = tar.getmembers()
    
    # Find all unique classes (directories) in the tar file
    class_names = list(set(os.path.dirname(m.name) for m in members if m.isfile()))
    
    # Calculate the number of classes to include (95%)
    num_classes_to_include = int(len(class_names) * 0.95)
    
    # Randomly select 95% of the classes
    selected_classes = set(random.sample(class_names, num_classes_to_include))

    # Iterate through each member in the tar file
    for member in members:
        if member.isfile():
            class_name = os.path.dirname(member.name)
            if class_name in selected_classes:
                # Extract the image into memory
                extracted_file = tar.extractfile(member)
                if extracted_file is not None:
                    # Create a new name for the image
                    image_name = os.path.basename(member.name)
                    new_image_name = f"{os.path.basename(class_name)}_{image_name}"
                    new_image_path = os.path.join(train_images_directory, new_image_name)
                    
                    # Write the image directly to the new path
                    with open(new_image_path, 'wb') as out_file:
                        shutil.copyfileobj(extracted_file, out_file)

print("Selected classes have been extracted, copied, and renamed successfully.")
