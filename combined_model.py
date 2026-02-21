import os
import cv2
import numpy as np
import io
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import uuid

def check_corresponding_pixels(image1_pil, image2_pil, bbox):
    """
    Checks if at least two corners in the bounding box have high pixel values in image2.
    """
    image1 = np.array(image1_pil)
    image2 = np.array(image2_pil)

    x_center, y_center, width, height = bbox
    x_min = int(max(0, x_center - width / 2))
    y_min = int(max(0, y_center - height / 2))
    x_max = int(min(image1.shape[1], x_center + width / 2))
    y_max = int(min(image1.shape[0], y_center + height / 2))

    if x_min >= x_max or y_min >= y_max:
        print("Warning: Bounding box is invalid or outside image1.")
        return 0, x_min, y_min, x_max, y_max

    corners = [
        [x_min, y_min],  
        [x_max, y_min],  
        [x_max, y_max],  
        [x_min, y_max]  
    ]

    
    num_high_value_corners = 0
    for corner in corners:
        x, y = corner
        if 0 <= y < image2.shape[0] and 0 <= x < image2.shape[1]:
            pixel_value = image2[y, x]
            if pixel_value[0] > 100 and pixel_value[1] > 100 and pixel_value[2] > 100:
                num_high_value_corners += 1

    return num_high_value_corners, x_min, y_min, x_max, y_max

def display_image(image_pil):
    """
    Displays an image using Matplotlib.
    """
    image = np.array(image_pil)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def process_image(model_path, img_input):
   
    model = YOLO(model_path)

   
    if hasattr(img_input, 'read'):  
        img_pil = Image.open(img_input)
    elif isinstance(img_input, str):  
        img_pil = Image.open(img_input)
    elif isinstance(img_input, Image.Image): 
        img_pil = img_input
    else:
        raise ValueError("Unsupported image input type")

    # Generate a unique filename for the temporary image
    temp_image_path = f"temp_image_{uuid.uuid4()}.jpg"
    img_pil.save(temp_image_path, format="JPEG")

    # Run inference on the image
    results = model(temp_image_path)
    boxes = results[0].boxes

    # Convert PIL image to OpenCV format
    image1 = np.array(img_pil)

    for box in boxes:
        bbox = box.xywh[0].tolist()  # Convert tensor to list
        num_high_value_corners, x_min, y_min, x_max, y_max = check_corresponding_pixels(img_pil, img_pil, bbox)
        if num_high_value_corners >= 2:
            print(f"Detected {num_high_value_corners} corners with high values in bounding box.")
            # Draw the bounding box on the image
            cv2.rectangle(image1, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    highlighted_image_pil = Image.fromarray(image1)

    # Clean up temporary image file
    os.remove(temp_image_path)

    return highlighted_image_pil

if __name__ == "__main__":
    MODEL_PATH = "path/to/your/model.pt"
    IMAGE_PATH = "path/to/your/image.jpg"

    # Open image directly using PIL
    img_pil = Image.open(IMAGE_PATH)
    
    # Process the image
    result_image = process_image(MODEL_PATH, img_pil)
    
    # Display the result
    display_image(result_image)