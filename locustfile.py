from locust import HttpUser, task, between
import os
from io import BytesIO

class ImageUser(HttpUser):
    wait_time = between(1, 3)  # Random wait time between requests
    
    @task
    def upload_image(self):
        # Read the image from the 'images' folder
        image_path = os.path.join(os.getcwd(), 'images', 'meher1.png')
        with open(image_path, 'rb') as img_file:
            img_byte_arr = img_file.read()
        
        # Simulate image upload in the /predict route
        self.client.post("/predict", files={"test_image": ("meher1.png", img_byte_arr, "image/png")})

    def on_start(self):
        """Called when a simulated user starts, you can log in or do pre-setup tasks here"""
        pass
