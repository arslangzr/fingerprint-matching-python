# Fingerprint Matching Project

This project demonstrates fingerprint matching using OpenCV's SIFT (Scale-Invariant Feature Transform) and FLANN (Fast Library for Approximate Nearest Neighbors) based matcher. The application compares a test fingerprint image with a set of fingerprint images stored in a database, identifying matches based on keypoint features.

## Requirements

- Python 3.12
- OpenCV
- NumPy
- Joblib

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/arslangzr/fingerprint-matching-python.git
cd fingerprint-matching-python
```

### 2. Set Up a Python Virtual Environment
Create a virtual environment using `python3` and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

### 3. Install Dependencies
Once the virtual environment is activated, install the required dependencies from `requirements.txt`:
```bash
pip3 install -r requirements.txt
```

### 4. Run the Application
After installing the dependencies, you can run the fingerprint matching application. Make sure that your test fingerprint image and the fingerprint database are in the appropriate folders as expected by the script.

To run the application, use the following command:
```bash
python fingerprint_detection.py
```

### 5. Ensure Input Files are Present
Make sure that:

- The test fingerprint image (e.g., `TEST_1.tif`) is located in the project directory.
- The fingerprint database directory contains the necessary fingerprint images for comparison.

### 6. Troubleshooting
- **Errors When Running:** If you encounter any errors, double-check that all dependencies are correctly installed and that your paths to the images are correct.
- **Missing Files:** Ensure that you have the `TEST_1.tif` image in the project directory and that the database folder contains the images you want to compare against.

### 7. Deactivate the Virtual Environment
After you are done using the application, deactivate the virtual environment by running:
```bash
deactivate
```

Following these steps will help you set up and run the Fingerprint Matching Project successfully!


### Summary of Changes
- Added a step to run the `fingerprint_detection.py` script.
- Included details on ensuring input files are present.
- Provided troubleshooting tips for common issues.
- Added a step for deactivating the virtual environment after use.

This comprehensive guide should help users smoothly set up and run the project!
