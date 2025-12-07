# Gravity - Multi-Country ID & Passport Data Extractor

A powerful Python-based web application that extracts structured data from national ID cards and passports using OCR and MRZ (Machine Readable Zone) parsing.

## Features

- 🔍 **Intelligent OCR**: Advanced image preprocessing with CLAHE, bilateral filtering, and adaptive thresholding
- 🌍 **Multi-Country Support**: Designed to handle various national ID formats and passports
- 📄 **MRZ Parsing**: Robust TD3 passport MRZ extraction with error correction
- 🎯 **High Accuracy**: Check digit validation and OCR error correction
- 🌐 **Web Interface**: Modern, user-friendly web UI for easy file uploads
- 📊 **JSON Output**: Clean, structured JSON responses

## Technology Stack

- **Backend**: FastAPI (Python)
- **OCR Engine**: Tesseract OCR with pytesseract
- **Image Processing**: OpenCV, PIL/Pillow
- **Frontend**: HTML/CSS/JavaScript

## Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.8+**
2. **Tesseract OCR**
   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

### 1. Clone or Navigate to the Project

```bash
cd /Users/mac/Downloads/Gravity
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

### Option 1: Using Python Directly

```bash
python app.py
```

### Option 2: Using Uvicorn

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The `--reload` flag enables auto-reload during development.

### Option 3: Custom Host/Port

```bash
uvicorn app:app --host 127.0.0.1 --port 5000
```

## Accessing the Application

Once the server is running, open your browser and navigate to:

```
http://localhost:8000
```

or

```
http://127.0.0.1:8000
```

## Usage

1. **Open the web interface** in your browser
2. **Select a country** from the dropdown (if applicable)
3. **Upload an ID card or passport image** (JPG, PNG, or PDF)
4. **Click "Extract Data"**
5. **View the extracted JSON data** in the results panel

## API Endpoints

### GET `/`
Returns the web interface (HTML page)

### POST `/process`
Processes an uploaded ID/passport image

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "mrz_found": true,
  "data": {
    "valid": true,
    "document_type": "P",
    "country_code": "CAN",
    "surname": "SMITH",
    "given_names": "JOHN DAVID",
    "full_name": "JOHN DAVID SMITH",
    "passport_number": "AB123456",
    "nationality": "CAN",
    "date_of_birth": "1990-05-15",
    "sex": "M",
    "expiry_date": "2030-12-31",
    "personal_number": null,
    "check_digits": {
      "passport_valid": true,
      "dob_valid": true,
      "expiry_valid": true,
      "final_valid": true
    },
    "raw_mrz": ["...", "..."]
  }
}
```

## Project Structure

```
Gravity/
├── app.py              # FastAPI application and routes
├── extractor.py        # Core OCR and MRZ parsing logic
├── index.html          # Web interface
├── requirements.txt    # Python dependencies
├── uploads/            # Temporary upload directory
├── test_samples/       # Sample test images
└── README.md           # This file
```

## Core Components

### `app.py`
- FastAPI web server
- File upload handling
- API endpoints

### `extractor.py`
- **PassportOCR**: Image preprocessing and OCR extraction
- **MRZParser**: MRZ parsing with check digit validation
- **Extractor**: Main facade for processing images

## Supported Document Types

- ✅ Passports (TD3 format with MRZ)
- ✅ National ID cards (various countries)
- ✅ Emirates ID (UAE)
- ✅ CNIC (Pakistan)
- 🔄 More countries being added

## Development

### Running in Development Mode

```bash
uvicorn app:app --reload --log-level debug
```

### Testing OCR Setup

To verify your Tesseract installation is working:

```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
```

## Troubleshooting

### Tesseract Not Found

If you get a "Tesseract not found" error:

1. Verify installation: `tesseract --version`
2. If installed but not found, set the path in your code:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/tesseract'
   ```

### Poor OCR Results

- Ensure images are clear and well-lit
- Try higher resolution images
- Check that the MRZ area is visible and not obscured

### Port Already in Use

If port 8000 is already in use:

```bash
uvicorn app:app --port 8001
```

## Dependencies

- `pytesseract==0.3.10` - Python wrapper for Tesseract OCR
- `opencv-python==4.8.1.78` - Image processing
- `pdf2image==1.16.3` - PDF to image conversion
- `Pillow==10.1.0` - Image manipulation
- `numpy==1.26.2` - Numerical operations
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `python-multipart==0.0.6` - File upload support

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is provided as-is for educational and development purposes.

## Notes

- Uploaded files are temporarily stored in the `uploads/` directory
- For production use, consider implementing file cleanup and security measures
- The application currently processes images synchronously; for high-volume use, consider async processing

## Contact & Support

For issues or questions, please open an issue in the project repository.
