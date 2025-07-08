import cv2
import numpy as np
from PIL import Image
import os

def crop_emojis_from_grid(image_path, output_dir="cropped_emojis"):
    """
    Crop individual emojis from a grid image and save them as separate PNG files.
    
    Args:
        image_path (str): Path to the input image containing the emoji grid
        output_dir (str): Directory to save the cropped emoji images
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to RGB (PIL format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to grayscale for grid detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect horizontal and vertical lines to find grid structure
    # Create kernels for detecting horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    
    # Detect horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    
    # Find contours to detect grid cells
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Alternative approach: Calculate grid dimensions based on image size
    # Assuming roughly equal-sized cells
    height, width = image.shape[:2]
    
    # Estimate number of rows and columns by analyzing the grid structure
    # For a typical emoji grid, we'll try to detect the pattern
    
    # Find grid lines by looking for consistent white/light areas
    # Calculate average cell size
    rows = 8  # Adjust based on your grid
    cols = 17  # Adjust based on your grid
    
    # Calculate cell dimensions
    cell_width = width // cols
    cell_height = height // rows
    
    emoji_count = 0
    
    # Extract each cell
    for row in range(rows):
        for col in range(cols):
            # Calculate cell boundaries
            x1 = col * cell_width
            y1 = row * cell_height
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            
            # Add small padding to avoid grid lines
            padding = 2
            x1 += padding
            y1 += padding
            x2 -= padding
            y2 -= padding
            
            # Skip if coordinates are invalid
            if x1 >= x2 or y1 >= y2:
                continue
                
            # Crop the emoji from the original image
            emoji_crop = pil_image.crop((x1, y1, x2, y2))
            
            # Check if the cropped area contains meaningful content
            # Convert to numpy array to analyze
            emoji_array = np.array(emoji_crop)
            
            # Skip if the cropped area is too small or mostly white/empty
            if emoji_array.size == 0:
                continue
                
            # Calculate variance to check if there's actual content
            gray_crop = cv2.cvtColor(emoji_array, cv2.COLOR_RGB2GRAY)
            variance = np.var(gray_crop)
            
            # Only save if there's enough variation (not just white space)
            if variance > 100:  # Adjust threshold as needed
                emoji_count += 1
                filename = f"emoji_{emoji_count:03d}.png"
                filepath = os.path.join(output_dir, filename)
                emoji_crop.save(filepath)
                print(f"Saved: {filename}")
    
    print(f"Total emojis extracted: {emoji_count}")

def crop_emojis_advanced_detection(image_path, output_dir="cropped_emojis_advanced"):
    """
    Advanced emoji cropping using contour detection and clustering.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    min_area = 500  # Adjust based on emoji size
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # Filter by reasonable aspect ratio for emojis
            if 0.5 <= aspect_ratio <= 2.0:
                valid_contours.append((x, y, w, h))
    
    # Sort contours by position (top to bottom, left to right)
    valid_contours.sort(key=lambda x: (x[1], x[0]))
    
    # Extract and save each emoji
    for i, (x, y, w, h) in enumerate(valid_contours):
        # Add some padding
        padding = 5
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop emoji
        emoji_crop = pil_image.crop((x1, y1, x2, y2))
        
        # Save
        filename = f"emoji_advanced_{i+1:03d}.png"
        filepath = os.path.join(output_dir, filename)
        emoji_crop.save(filepath)
        print(f"Saved: {filename}")
    
    print(f"Total emojis extracted (advanced): {len(valid_contours)}")

if __name__ == "__main__":
    # Usage example
    image_path = "emojis.png"  # Replace with your image path
    
    print("Method 1: Grid-based cropping")
    crop_emojis_from_grid(image_path)
    
    print("\nMethod 2: Contour-based cropping")
    crop_emojis_advanced_detection(image_path)
    
    print("\nDone! Check the output directories for cropped emojis.")