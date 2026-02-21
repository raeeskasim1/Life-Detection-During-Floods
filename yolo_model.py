import os
from ultralytics import YOLO
from PIL import Image
import io
import uuid

def run_yolo_inference(model_path, image_input):
    
    model = YOLO(model_path)

    
    if hasattr(image_input, 'read'):  
        img_pil = Image.open(image_input)
    elif isinstance(image_input, str): 
        img_pil = Image.open(image_input)
    elif isinstance(image_input, Image.Image):  
        img_pil = image_input
    else:
        raise ValueError("Unsupported image input type")

    
    temp_image_path = f"temp_image_{uuid.uuid4()}.jpg"
    img_pil.save(temp_image_path, format="JPEG")

    
    results = model(temp_image_path)

    
    save_dir = os.path.join(os.getcwd(), "predictions")
    os.makedirs(save_dir, exist_ok=True)

    saved_image_path = None
    if results:
        
        result = results[0]
        saved_image_path = os.path.join(save_dir, f"result_{uuid.uuid4()}.jpg")
        result.save(saved_image_path)  

    
    os.remove(temp_image_path)

    return result.boxes if results else [], saved_image_path

if __name__ == "__main__":
    MODEL_PATH = "C:/Users/HP/OneDrive/Desktop/finall/models/yolov8/best.pt"
    IMAGE_PATH = "C:/Users/HP/OneDrive/Desktop/finall/sample_data/boat.jpg"

    
    img_pil = Image.open(IMAGE_PATH)
    
    boxes, saved_image_path = run_yolo_inference(MODEL_PATH, img_pil)
    

    print("Detected boxes:")
    for box in boxes:
        print(box)

    
    if saved_image_path:
        result_image_pil = Image.open(saved_image_path)
        result_image_pil.show()
    else:
        print("No result image saved.")