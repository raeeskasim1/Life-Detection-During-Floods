import streamlit as st
from PIL import Image
import io
import torch
import os 
import cv2
import tempfile
from unet_model import single_image_inference
from yolo_model import run_yolo_inference
from combined_model import process_image


st.title("Computer Vision Guided Life Detection During Floods")


uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "png", "mp4", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == "image":
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        image_pil = Image.open(uploaded_file)
        process_media = image_pil
    elif file_type == "video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        st.video(video_path)
        process_media = video_path
    else:
        st.error("Unsupported file type")
        st.stop()

    
    model_options = st.sidebar.selectbox("Choose model to run:", ["Segmentation", "Object Detection", "Final Output"])

    if model_options == "Segmentation":
        st.sidebar.text("Running UNet model...")
        with st.spinner("Processing..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_pth = "C:/Users/RAEES/Desktop/prolife/models/unet_test1_15.pth"  
            if file_type == "image":
                single_image_inference(model_pth, uploaded_file, device)
                st.image("output.jpg", caption="UNet Output")
            else:
               
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 30 == 0:  
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        single_image_inference(model_pth, frame_pil, device)
                        st.image("output.jpg", caption=f"UNet Output - Frame {frame_count}")
                cap.release()

    elif model_options == "Object Detection":
        st.sidebar.text("Running YOLO model...")
        with st.spinner("Processing..."):
            model_path = "C:/Users/RAEES/Desktop/prolife/models/yolov8/best.pt"  
            if file_type == "image":
                _, saved_image_path = run_yolo_inference(model_path, uploaded_file)
                if saved_image_path and os.path.exists(saved_image_path):
                    st.image(saved_image_path, caption="YOLO Output", use_column_width=True)
                else:
                    st.write("No result image available or failed to save result.")
            else:
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 30 == 0:  
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        _, saved_image_path = run_yolo_inference(model_path, frame_pil)
                        if saved_image_path and os.path.exists(saved_image_path):
                            st.image(saved_image_path, caption=f"YOLO Output - Frame {frame_count}", use_column_width=True)
                cap.release()

    elif model_options == "Final Output":
        st.sidebar.text("Running Combined model...")
        with st.spinner("Processing..."):
            model_path = "C:/Users/RAEES/Desktop/prolife/models/yolov8/best.pt"  
            if file_type == "image":
                highlighted_image_pil = process_image(model_path, image_pil)
                highlighted_image_pil.save("highlighted_image.jpg")
                st.image("highlighted_image.jpg", caption="Integrated Result")
            else:

                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % 30 == 0:  
                        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        highlighted_image_pil = process_image(model_path, frame_pil)
                        highlighted_image_pil.save(f"highlighted_image_{frame_count}.jpg")
                        st.image(f"highlighted_image_{frame_count}.jpg", caption=f"Integrated Result - Frame {frame_count}")
                cap.release()

else:
    st.text("Please upload an image or video file.")