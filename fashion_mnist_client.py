import gradio as gr
import requests
import io
import pandas as pd
import numpy as np
from PIL import Image

def get_image_from_csv(csv_path, index=0):
    df = pd.read_csv(csv_path)
    row = df.iloc[index]
    label = int(row['label'])
    pixels = row.drop('label').to_numpy().astype(np.uint8)
    image_array = pixels.reshape(28, 28)
    image = Image.fromarray(image_array, mode='L')
    return image, label

csv_path = r"C:\shsAI\fashion_mnist\fashion-mnist_test.csv"

def classify_from_csv_row(index):
    image, actual_label = get_image_from_csv(csv_path, index)
    
    url = "http://127.0.0.1:8000/classify"
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes = image_bytes.getvalue()
    
    response = requests.post(url, files={"file": ("image.png", image_bytes, "image/png")})
    if response.status_code == 200:
        predicted_label = response.json().get("label", "Error")
    else:
        predicted_label = "Error"
    
    return image, f"예측: {predicted_label} | 실제: {actual_label}"


iface = gr.Interface(
    fn=classify_from_csv_row,
    inputs=gr.Slider(minimum=0, maximum=9999, step=1, label="테스트 이미지 인덱스 선택"),
    outputs=[gr.Image(), "text"],
    title="FashionMNIST 이미지 분류기",
    description="FashionMNIST 테스트 이미지 분류"
)

if __name__ == "__main__":
    iface.launch()
