
import os
import shutil
import tarfile

# Define the path to your .tar.gz file
tar_path = 'rwave-1024.tar.gz'
extract_path = './'

# Extract the tar.gz file
with tarfile.open(tar_path, 'r:gz') as tar:
    tar.extractall(path=extract_path)

# Path to the main directory containing class folders
main_directory = 'images'
# Path to the directory where you want to pool all images
train_images_directory = 'train_images'

# Create the train_images directory if it doesn't exist
os.makedirs(train_images_directory, exist_ok=True)

# Iterate through each class directory
for class_name in os.listdir(main_directory):
    class_path = os.path.join(main_directory, class_name)
    
    # Check if it's a directory
    if os.path.isdir(class_path):
        # Iterate through each image in the class directory
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            # Create the new image name
            new_image_name = f"{class_name}_{image_name}n"
            new_image_path = os.path.join(train_images_directory, new_image_name)
            
            # Copy the image to the train_images directory with the new name
            shutil.copy2(image_path, new_image_path)

print("All images have been copied and renamed successfully.")
