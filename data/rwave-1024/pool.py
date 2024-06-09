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
    image_members = [m for m in members if m.isfile() and m.name.endswith(('.jpg', '.jpeg', '.png'))]

    # Calculate the number of images to include (95%)
    num_images_to_include = int(len(image_members) * 0.95)

    # Randomly select 95% of the images
    selected_image_members = random.sample(image_members, num_images_to_include)

    # Extract and rename the selected images
    for member in selected_image_members:
        # Extract the image into memory
        extracted_file = tar.extractfile(member)
        if extracted_file is not None:
            # Create a new name for the image
            class_name = os.path.basename(os.path.dirname(member.name))
            image_name = os.path.basename(member.name)
            new_image_name = f"{class_name}_{image_name}"
            new_image_path = os.path.join(train_images_directory, new_image_name)
            
            # Write the image directly to the new path
            with open(new_image_path, 'wb') as out_file:
                shutil.copyfileobj(extracted_file, out_file)

print("Selected images have been extracted, copied, and renamed successfully.")
