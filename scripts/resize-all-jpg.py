import os
from PIL import Image

def resize_images_in_directory(directory, scale):
    """
    Reduces the size of all JPG files in a directory by a given scale.
    
    :param directory: The path to the directory containing the images.
    :param scale: The scale by which to reduce the image size (e.g., 0.5 to reduce size by half).
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"The directory {directory} does not exist.")
        return
    
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            img_path = os.path.join(directory, filename)
            try:
                # Open the image
                with Image.open(img_path) as img:
                    # Calculate new dimensions
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                    
                    # Resize the image
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save the resized image (overwriting the original file)
                    resized_img.save(img_path)
                    print(f"Resized {filename} to {new_width}x{new_height} pixels.")
            except Exception as e:
                print(f"Failed to resize {filename}: {e}")

if __name__ == "__main__":
    directory = os.path.expanduser('~/Desktop/gold2/')
    scale = 0.5  # Example: reduce size by 50%
    resize_images_in_directory(directory, scale)
