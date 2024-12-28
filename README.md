### Integrating YOLO11 and Tesseract OCR for Object Detection and Text Recognition

Integrating object detection with YOLO and Optical Character Recognition (OCR) using Tesseract. The goal is to identify specific regions in images (e.g., invoices) using YOLO and extract text from those regions using OCR. We'll go step by step, from setting up the environment to implementing the solution.

Read the full article here

![screenshot](Integrating YOLO11 and Tesseract OCR for Object Detection and Text Recognition.png)
---

### Step 1: **Setting Up the Environment**

#### Install Required Libraries

To start, ensure you have the necessary libraries and tools installed:

1. **Install YOLOv8**:
   YOLOv8 is part of the `ultralytics` package, which provides powerful tools for object detection and segmentation.

   ```bash
   pip install ultralytics
   ```

2. **Install OpenCV**:
   OpenCV is used for image processing.

   ```bash
   pip install opencv-python
   ```

3. **Install Pytesseract**:
   Pytesseract acts as a Python wrapper for Tesseract OCR.

   ```bash
   pip install pytesseract
   ```

4. **Install Tesseract OCR**:
   Download and install Tesseract OCR from the [official repository](https://github.com/UB-Mannheim/tesseract/wiki) if you're on Windows, or use your package manager for Linux:

   ```bash
   sudo apt install tesseract-ocr  # For Ubuntu
   ```

   On Windows, add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`) to your system's PATH.

---

### Step 2: **Loading the Trained YOLO Model**

Assume you've trained a YOLO model to detect regions of interest in your images, such as specific fields in invoices.

Here’s how to load the trained YOLO model:

```python
from ultralytics import YOLO

# Load the best-trained YOLO model
model = YOLO('path/to/your/best.pt')
```

The model is now ready to detect objects in images.

---

### Step 3: **Defining the OCR Workflow**

#### OCR Function

Define a function to crop detected regions and extract text using Tesseract OCR.

```python
import cv2
import pytesseract

def perform_ocr(image, detections):
    """
    Perform OCR on cropped regions from the detected bounding boxes and include class names.
    """
    for i, detection in enumerate(detections):
        # Extract bounding box and class name
        x1, y1, x2, y2, class_name = detection
        cropped_image = image[int(y1):int(y2), int(x1):int(x2)]

        # Preprocess for better OCR results
        gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Perform OCR using Tesseract
        text = pytesseract.image_to_string(binary, lang='eng')
        print(f"Class '{class_name}' detected: {text.strip()}")

        # Optional: Display cropped region
        cv2.imshow(f"Region - {class_name}", binary)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
```

This function:

- Crops detected regions.
- Converts them to grayscale and applies thresholding for better OCR results.
- Extracts text using Tesseract.

---

### Step 4: **Processing Images**

Process each image in a folder, run the YOLO model, and apply the OCR function.

```python
import os

# Path to images folder
images_folder = 'path/to/images/folder'

# Iterate through images
for file_name in os.listdir(images_folder):
    if file_name.endswith(('.jpg', '.jpeg', '.png')):  # Check for image files
        file_path = os.path.join(images_folder, file_name)

        # Load the image
        image = cv2.imread(file_path)

        # Run YOLO model
        results = model(file_path)

        # Extract bounding boxes and class names
        detections = []
        for box in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box[:6]
            class_name = results[0].names[int(cls)]  # Map class index to class name
            detections.append((x1, y1, x2, y2, class_name))

        print(f"Detected objects for {file_name}: {detections}")

        # Perform OCR
        perform_ocr(image, detections)
```

This script:

1. Iterates over all image files in a directory.
2. Detects objects using YOLO.
3. Passes detected regions to the OCR function.

---

### Step 5: **Fine-Tuning OCR**

#### Preprocessing for Better OCR Results

Preprocessing is crucial to improve OCR accuracy. Use techniques like:

- **Grayscale Conversion**:
  Converts the image to grayscale for easier text recognition.
  ```python
  gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
  ```
- **Thresholding**:
  Enhances text visibility by binarizing the image.
  ```python
  _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
  ```

#### Using Language-Specific OCR

Specify the language in Tesseract using the `lang` parameter (e.g., English: `lang='eng'`).

---

### Step 6: **Results**

For each image, the script:

- Detects objects (e.g., `invoice_number`, `date`, etc.).
- Extracts text from the detected regions.
- Displays the extracted text along with the class name.

#### Sample Output

For an invoice image with detected regions:

```
Detected objects for invoice1.jpg: [(50, 100, 200, 150, 'invoice_number'), (300, 400, 500, 450, 'date')]
Class 'invoice_number' detected: INV-2024-00123
Class 'date' detected: 11/25/2024
```

---

### Conclusion

This step-by-step pipeline combines the power of YOLO for object detection with Tesseract for OCR. The solution can be applied to various use cases, such as:

- Automating data extraction from invoices, receipts, or documents.
- Analyzing text within detected regions in images.

The flexibility of YOLO and Tesseract ensures this pipeline can adapt to diverse applications. Experiment with different preprocessing techniques and model configurations to optimize performance for your specific dataset.
