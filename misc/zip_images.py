import os
import zipfile
from datetime import datetime

def zip_images(source_dir, output_dir=None, image_extensions=('.png', '.jpg', '.jpeg')):
    """
    Zip all images from the source directory.
    
    Args:
        source_dir (str): Directory containing the images to zip
        output_dir (str, optional): Directory to save the zip file. If None, uses source_dir
        image_extensions (tuple): Tuple of image file extensions to include
    """
    # If no output directory specified, use source directory
    if output_dir is None:
        output_dir = source_dir
    
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    zip_filename = f'images_{timestamp}.zip'
    zip_path = os.path.join(output_dir, zip_filename)
    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                # Check if file is an image
                if file.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file)
                    # Get the relative path for the file in the zip
                    rel_path = os.path.relpath(file_path, source_dir)
                    # Add file to zip
                    zipf.write(file_path, rel_path)
                    print(f'Added {file} to zip')
    
    print(f'\nZip file created successfully: {zip_path}')
    print(f'Total files zipped: {len(zipf.namelist())}')

if __name__ == '__main__':
    # Example usage
    source_directory = './images/test_results'  # Change this to your images directory
    output_directory = './images'  # Change this to where you want the zip file
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Zip the images
    zip_images(source_directory, output_directory) 