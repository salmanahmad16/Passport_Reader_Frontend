"""
Core Extraction Logic
Contains PassportOCR and MRZParser classes for extracting data from IDs.
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Tuple, List, Dict, Optional
import re

# ==========================================
# PassportOCR Class
# ==========================================

class PassportOCR:
    """OCR processor for passport images"""
    
    def __init__(self, tesseract_config: str = '--oem 3 --psm 6'):
        """
        Initialize OCR processor
        
        Args:
            tesseract_config: Tesseract configuration string
                --oem 3: Use default OCR Engine Mode (LSTM)
                --psm 6: Assume uniform block of text
        """
        self.tesseract_config = tesseract_config
        self.mrz_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter - reduce noise, keep edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)
        
        # Slight Gaussian blur
        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def detect_mrz_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect and extract MRZ region from passport image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        morph = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        height, width = gray.shape
        mrz_candidates = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            
            # MRZ is at bottom, high aspect ratio
            if aspect_ratio > 5 and y > height * 0.5 and w > width * 0.5:
                mrz_candidates.append((y, x, w, h))
        
        if mrz_candidates:
            mrz_candidates.sort(reverse=True)
            y, x, w, h = mrz_candidates[0]
            
            padding = 10
            y1 = max(0, y - padding)
            y2 = min(height, y + h + padding)
            x1 = max(0, x - padding)
            x2 = min(width, x + w + padding)
            
            return gray[y1:y2, x1:x2]
        
        # Fallback: return bottom portion
        return gray[int(height * 0.7):, :]
    
    def extract_text(self, image: np.ndarray, config: Optional[str] = None) -> str:
        """Extract text using Tesseract"""
        if config is None:
            config = self.tesseract_config
        
        processed = self.preprocess_image(image)
        pil_image = Image.fromarray(processed)
        return pytesseract.image_to_string(pil_image, config=config)
    
    def extract_mrz(self, image: np.ndarray) -> List[str]:
        """Extract MRZ lines"""
        mrz_region = self.detect_mrz_region(image)
        if mrz_region is None:
            return []
        
        processed = self.preprocess_image(mrz_region)
        pil_image = Image.fromarray(processed)
        mrz_text = pytesseract.image_to_string(pil_image, config=self.mrz_config)
        
        lines = []
        for line in mrz_text.split('\n'):
            cleaned = line.strip().replace(' ', '')
            if len(cleaned) >= 30:
                valid_chars = sum(1 for c in cleaned if c.isalnum() or c == '<')
                if valid_chars / len(cleaned) > 0.8:
                    lines.append(cleaned)
        return lines
    
    def extract_all_text(self, image: np.ndarray) -> Dict[str, any]:
        """Extract MRZ and full text"""
        mrz_lines = self.extract_mrz(image)
        full_text = self.extract_text(image)
        
        return {
            'mrz_lines': mrz_lines,
            'full_text': full_text,
            'mrz_found': len(mrz_lines) >= 2
        }

    def process_image_file(self, image_path: str) -> Dict[str, any]:
        """Process image file from path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        return self.extract_all_text(image)


# ==========================================
# MRZParser Class
# ==========================================

class MRZParser:
    """Parser for passport MRZ data (TD3 format)"""
    
    OCR_CORRECTIONS = {
        'O': '0', 'I': '1', 'S': '5', 'B': '8', 'Z': '2',
    }
    
    def clean_mrz_line(self, line: str) -> str:
        """Clean and normalize MRZ line"""
        line = line.strip().upper()
        # Remove spaces
        line = line.replace(' ', '')
        
        # Replace padded K/5 with <
        line = re.sub(r'[K5]{3,}', lambda m: '<' * len(m.group()), line)
        # Remove invalid chars
        line = re.sub(r'[^A-Z0-9<]', '', line)
        
        return line
    
    def calculate_check_digit(self, data: str) -> int:
        """Calculate MRZ check digit"""
        weights = [7, 3, 1]
        total = 0
        for i, char in enumerate(data):
            if char == '<':
                value = 0
            elif char.isdigit():
                value = int(char)
            else:
                value = ord(char) - ord('A') + 10
            total += value * weights[i % 3]
        return total % 10
    
    def validate_check_digit(self, data: str, check_digit: str) -> bool:
        """Validate check digit"""
        try:
            expected = self.calculate_check_digit(data)
            actual = int(check_digit)
            return expected == actual
        except (ValueError, IndexError):
            return False
            
    def correct_check_digit_field(self, data: str, check_digit: str) -> Tuple[str, str]:
        """Attempt correction of check digit fields"""
        if self.validate_check_digit(data, check_digit):
            return data, check_digit
            
        # Try correcting check digit itself
        for wrong, correct in self.OCR_CORRECTIONS.items():
            if wrong in check_digit:
                corrected_digit = check_digit.replace(wrong, correct)
                if self.validate_check_digit(data, corrected_digit):
                    return data, corrected_digit
                    
        # Try correcting data field
        for i, char in enumerate(data):
            if char in self.OCR_CORRECTIONS:
                corrected_data = data[:i] + self.OCR_CORRECTIONS[char] + data[i+1:]
                if self.validate_check_digit(corrected_data, check_digit):
                    return corrected_data, check_digit
                    
        return data, check_digit

    def _clean_name_ocr_errors(self, name: str) -> str:
        """Remove trailing OCR artifacts from names"""
        if not name: return name
        
        # Common trailing errors
        if len(name) >= 2:
            suffix_2 = name[-2:]
            if suffix_2 in ['SS', 'RS', 'DS', 'NS', 'KK']:
                name = name[:-1]
        
        if name and name[-1] == '5':
            name = name[:-1]
            
        return name.strip()

    def parse_name(self, name_field: str) -> Dict[str, str]:
        """Parse name into surname and given names"""
        cleaned = name_field
        
        # Replace padding noise
        cleaned = re.sub(r'K{3,}', lambda m: '<' * len(m.group()), cleaned)
        cleaned = re.sub(r'5{3,}', lambda m: '<' * len(m.group()), cleaned)
        cleaned = re.sub(r'[K5]{3,}', lambda m: '<' * len(m.group()), cleaned)
        
        parts = cleaned.split('<<')
        
        surname = ''
        given_names = ''
        
        # Handle severe corruption
        if len(parts) >= 1:
            first_part = parts[0].replace('<', '').strip()
            
            # Check for valid given names in part 2
            has_valid_given = False
            if len(parts) >= 2:
                given_part = parts[1].replace('<', '').replace('K', '').replace('5', '').strip()
                if len(given_part) > 0 and any(c.isalpha() for c in given_part):
                    has_valid_given = True
            
            # If deeply corrupted (long first part, no valid given names)
            if len(first_part) > 15 and ' ' not in first_part and not has_valid_given:
                # Heuristic split
                first_part = re.sub(r'[0-9]+', '', first_part)
                split_point = 7
                # Search for surname ending patterns E,N,D,R
                for i in range(9, 5, -1):
                    if i < len(first_part) and first_part[i] in 'ENDR':
                        if i + 1 < len(first_part):
                            split_point = i + 1
                            break
                            
                surname = first_part[:split_point]
                given_names_raw = first_part[split_point:]
                
                # Add spaces
                given_names = re.sub(r'([A-Z]{3,})([A-Z]{3,})', r'\1 \2', given_names_raw)
                
                surname = self._clean_name_ocr_errors(surname)
                given_names = self._clean_name_ocr_errors(given_names)
                
                return {
                    'surname': surname,
                    'given_names': given_names,
                    'full_name': f"{given_names} {surname}".strip(),
                    'warning': 'Names extracted from corrupted MRZ'
                }
        
        # Normal Parsing
        if len(parts) >= 1:
            surname = parts[0].replace('<', ' ').strip()
            surname = self._clean_name_ocr_errors(surname)
            
        if len(parts) >= 2:
            given_part = parts[1]
            given_part = re.sub(r'[<K5]+$', '', given_part)
            given_names = given_part.replace('<', ' ').strip()
            given_names = self._clean_name_ocr_errors(given_names)
            
        return {
            'surname': surname,
            'given_names': given_names,
            'full_name': f"{given_names} {surname}".strip()
        }

    def parse_date(self, date_str: str) -> str:
        """YYMMDD -> YYYY-MM-DD"""
        if len(date_str) != 6 or not date_str.isdigit():
            return date_str
        
        year = int(date_str[0:2])
        month = date_str[2:4]
        day = date_str[4:6]
        
        full_year = f"20{year:02d}" if year <= 30 else f"19{year:02d}"
        return f"{full_year}-{month}-{day}"

    def normalize_mrz_line(self, line: str, line_number: int) -> str:
        """Ensure line is 44 chars"""
        current_len = len(line)
        if current_len == 44: return line
        
        if current_len > 44:
            excess = current_len - 44
            if line_number == 1:
                return line[:44] # Trim end
            else:
                # Trim from personal number area (28-42) usually
                return line[:28] + line[28+excess:44+excess] + line[44+excess:]
        else:
            return line + ('<' * (44 - current_len))

    def parse_td3(self, line1: str, line2: str) -> Dict[str, any]:
        """Parse TD3 2-line MRZ"""
        line1 = self.clean_mrz_line(line1)
        line2 = self.clean_mrz_line(line2)
        
        quality_issues = []
        
        if len(line1) != 44 or len(line2) != 44:
            line1 = self.normalize_mrz_line(line1, 1)
            line2 = self.normalize_mrz_line(line2, 2)
            quality_issues.append("Line length correction applied")

        # Basic validation
        if len(line1) != 44 or len(line2) != 44:
            return {'valid': False, 'error': 'Invalid MRZ length', 'quality_warning': 'Poor Image Quality'}

        # Line 1
        document_type = line1[0:2]
        country_code = line1[2:5].replace('<', '')
        name_field = line1[5:44]
        name_data = self.parse_name(name_field)
        
        if 'warning' in name_data:
            quality_issues.append("Name field corrupted")

        # Line 2
        passport_number = line2[0:9].replace('<', '')
        passport_check = line2[9]
        nationality = line2[10:13].replace('<', '')
        dob = line2[13:19]
        dob_check = line2[19]
        sex = line2[20]
        expiry_date = line2[21:27]
        expiry_check = line2[27]
        personal_number = line2[28:42].replace('<', '')
        personal_check = line2[42]
        final_check = line2[43]
        
        # Validations
        passport_number, passport_check = self.correct_check_digit_field(line2[0:9], passport_check)
        passport_number = passport_number.replace('<', '')
        dob, dob_check = self.correct_check_digit_field(dob, dob_check)
        expiry_date, expiry_check = self.correct_check_digit_field(expiry_date, expiry_check)
        
        passport_valid = self.validate_check_digit(line2[0:9], passport_check)
        dob_valid = self.validate_check_digit(dob, dob_check)
        expiry_valid = self.validate_check_digit(expiry_date, expiry_check)
        
        composite_data = line2[0:10] + line2[13:20] + line2[21:43]
        final_valid = self.validate_check_digit(composite_data, final_check)
        
        failures = sum([not passport_valid, not dob_valid, not expiry_valid, not final_valid])
        if failures >= 2:
            quality_issues.append(f"{failures} check digit failures")
            
        quality_warning = None
        if len(quality_issues) >= 1:
            quality_warning = "Potential OCR errors detected. verify data."

        parsed = {
            'valid': True,
            'document_type': document_type,
            'country_code': country_code,
            'surname': name_data['surname'],
            'given_names': name_data['given_names'],
            'full_name': name_data['full_name'],
            'passport_number': passport_number,
            'nationality': nationality,
            'date_of_birth': self.parse_date(dob),
            'sex': sex,
            'expiry_date': self.parse_date(expiry_date),
            'personal_number': personal_number if personal_number else None,
            'check_digits': {
                'passport_valid': passport_valid,
                'dob_valid': dob_valid,
                'expiry_valid': expiry_valid,
                'final_valid': final_valid
            },
            'raw_mrz': [line1, line2]
        }
        
        if quality_warning:
            parsed['quality_warning'] = quality_warning
            parsed['quality_issues'] = quality_issues
            
        return parsed

    def parse(self, mrz_lines: list) -> Dict[str, any]:
        """Parse list of MRZ lines"""
        if len(mrz_lines) < 2:
            return {'valid': False, 'error': 'Insufficient MRZ lines'}
            
        return self.parse_td3(mrz_lines[-2], mrz_lines[-1])


# ==========================================
# Main Facade
# ==========================================

class Extractor:
    def __init__(self):
        self.ocr = PassportOCR()
        self.parser = MRZParser()
        
    def process(self, image_path: str) -> Dict[str, any]:
        try:
            # OCR
            ocr_result = self.ocr.process_image_file(image_path)
            
            result = {
                'success': False,
                'mrz_found': ocr_result['mrz_found'],
                'raw_text_snippet': ocr_result['full_text'][:200] + '...' if ocr_result['full_text'] else ''
            }
            
            # Parse
            if ocr_result['mrz_found']:
                parsed = self.parser.parse(ocr_result['mrz_lines'])
                result['data'] = parsed
                result['success'] = parsed.get('valid', False)
            else:
                 result['error'] = "No MRZ found in image"
                 
            return result
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Simple global instance or function for easy import
_extractor = Extractor()
def extract_data(image_path: str) -> Dict[str, any]:
    return _extractor.process(image_path)
