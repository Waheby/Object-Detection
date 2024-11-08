#GENERAL API REQUEST IMPORTS
import cloudinary.uploader
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

from ultralytics import YOLO

import os
from dotenv import load_dotenv
load_dotenv()

# Cloudinary Configuration
import cloudinary
import cloudinary.uploader
import cloudinary.api
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True)


# Create a Flask application instance
app = Flask(__name__)
# Enable CORS for all routes, allowing requests from any origin
CORS(app,resources={r"/*":{"origins":"*"}})

# Define a route for HTTP request
@app.route('/validate', methods=['POST'])
def detect_certificate_objects():
    try:
        # Receive Data from Frontend API Request
        data = request.get_json()
        print(data['data'])
        print(data['is_export'])
        prediction = False
        url = None
        # Load my newly created model
        model = YOLO("best.pt") 

        # Run batched inference on a list of images
        results = model([data['data']])

        # Process results list
        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs

            if data['is_export'] == "True":
                result.save(filename="result.jpg")  # save to disk
                cloudinary.uploader.destroy("result")
                print("Uploading...")
                response = cloudinary.uploader.upload("result.jpg", public_id="result", unique_filename = False, overwrite=True, invalidate=True)
                url = response['url']

            if boxes:
                print("Valid Cert")
                prediction = True
            else:
                print("Invalid Cert")
                prediction = False
        
        return jsonify({'yoloPrediction': prediction, 'url': url if url != None else 'null'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5050)
    
    