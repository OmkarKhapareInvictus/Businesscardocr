import os
import re
import warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
warnings.filterwarnings("ignore", message=".*torch.quantize_per_tensor.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")

import cv2
import easyocr
import pandas as pd

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

CSV_FILE = "contacts_data.csv"

def find_best_orientation(img):
    """
    Tests 0°, 90°, 180°, and 270° rotations.
    Selects the orientation that produces the most valid alphabetical words.
    """
    rotations = [
        (0, img),
        (90, cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(img, cv2.ROTATE_180)),
        (270, cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    ]

    best_angle = 0
    best_image = img
    best_results = []
    max_valid_words = -1

    for angle, rotated_mat in rotations:
        # Quick OCR pass to check word validity
        results = reader.readtext(rotated_mat, detail=1, paragraph=False)
        
        # Count words with 3 or more letters (rejects single digits/symbols)
        valid_words = 0
        for _, text, conf in results:
            clean = re.sub(r'[^a-zA-Z]', '', text)
            if len(clean) >= 3 and conf > 0.2:
                valid_words += 1

        if valid_words > max_valid_words:
            max_valid_words = valid_words
            best_angle = angle
            best_image = rotated_mat
            best_results = results

    print(f"[INFO] Auto-detected best orientation: {best_angle}° rotation")
    return best_image, best_results

def extract_entities(text_lines):
    full_text = " \n ".join(text_lines)

    # 1. Email Address
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = re.findall(email_pattern, full_text)
    email = emails[0].lower() if emails else "N/A"

    # 2. Phone / Mobile Number
    phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}'
    raw_phones = re.findall(phone_pattern, full_text)
    valid_phones = [p.strip() for p in raw_phones if len(re.sub(r'\D', '', p)) >= 10]
    phone = valid_phones[0] if valid_phones else "N/A"

    # 3. Keyword Dictionaries
    position_keywords = [
        'engineer', 'manager', 'developer', 'director', 'lead', 'consultant',
        'officer', 'executive', 'founder', 'ceo', 'cto', 'associate', 'analyst', 'president'
    ]
    company_keywords = [
        'ltd', 'limited', 'pvt', 'inc', 'corp', 'solutions', 'tech', 'services', 
        'systems', 'group', 'co.', 'llc', 'studios', 'enterprises', 'invictus'
    ]
    address_keywords = [
        'street', 'road', 'floor', 'plot', 'city', 'state', 'pin', 'near', 
        'opp', 'lane', 'avenue', 'nagar', 'sector', 'building'
    ]

    name = "N/A"
    position = "N/A"
    company = "N/A"
    address_lines = []

    cleaned = [line.strip() for line in text_lines if len(line.strip()) >= 2]

    for line in cleaned:
        line_lower = line.lower()

        if email != "N/A" and email in line_lower:
            continue
        if phone != "N/A" and re.sub(r'\D', '', phone) in re.sub(r'\D', '', line):
            continue

        if position == "N/A" and any(k in line_lower for k in position_keywords):
            position = line
            continue

        if company == "N/A" and any(k in line_lower for k in company_keywords):
            company = line
            continue

        if any(k in line_lower for k in address_keywords) or (re.search(r'\b\d{5,6}\b', line) and len(line) > 8):
            address_lines.append(line)
            continue

        # Name heuristic: clean alphabetic string, 2-4 words, no numbers
        if name == "N/A" and not re.search(r'\d', line):
            words = line.split()
            if 1 <= len(words) <= 4 and all(w.replace('.', '').isalpha() for w in words):
                name = line

    return {
        "Name": name,
        "Position": position,
        "Mobile No": phone,
        "Company Name": company,
        "Email": email,
        "Address": ", ".join(address_lines) if address_lines else "N/A"
    }

def process_card(image_path):
    if not os.path.isfile(image_path):
        print(f"[ERROR] Cannot find file at: {image_path}")
        return

    # Load image
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"[ERROR] Could not decode image at: {image_path}")
        return

    # Resize up if resolution is low
    h, w = raw_img.shape[:2]
    if max(h, w) < 1400:
        scale = 1400.0 / max(h, w)
        raw_img = cv2.resize(raw_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # 1. Automatically test and set correct orientation
    best_img, ocr_results = find_best_orientation(raw_img)

    # Extract detected text lines
    extracted_lines = [item[1].strip() for item in ocr_results if item[2] > 0.25 and len(item[1].strip()) > 1]

    print("\n--- Extracted Text ---")
    for line in extracted_lines:
        print(f"• {line}")

    # 2. Parse into structured data
    card_data = extract_entities(extracted_lines)

    print("\n--- Structured Card Info ---")
    for k, v in card_data.items():
        print(f"{k:15}: {v}")

    # 3. Save to CSV
    df = pd.DataFrame([card_data])
    df.to_csv(CSV_FILE, mode='a', header=not os.path.exists(CSV_FILE), index=False)
    print(f"\n[SUCCESS] Output saved to '{CSV_FILE}'!")

if __name__ == "__main__":
    process_card("invictuscard.jpeg")
