import sys
try:
    print("Testing imports...")
    import fastapi
    import uvicorn
    import cv2
    import pytesseract
    import numpy
    import PIL
    
    from app import app
    from extractor import extract_data
    
    print("SUCCESS: All modules imported and app initialized.")
except Exception as e:
    print(f"FAILURE: {e}")
    sys.exit(1)
