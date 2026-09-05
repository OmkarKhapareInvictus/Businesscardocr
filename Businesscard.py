import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration & Paths ---
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

CSV_FILE = "scanned_addresses.csv"
IMAGE_DIR = "scanned_images"
BURST_FRAME_COUNT = 30  # 30 frames provides sufficient temporal denoising with less latency
SIMILARITY_THRESHOLD = 0.45  # 45% TF-IDF threshold for matching duplicate addresses
CSV_COLUMNS = ["Serial_No", "Address", "Status"]

os.makedirs(IMAGE_DIR, exist_ok=True)


def get_next_serial_number(file_path=CSV_FILE):
    """Determines the next consecutive numeric Serial Number from CSV."""
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if "Serial_No" in df.columns and len(df) > 0:
                valid_ids = pd.to_numeric(df["Serial_No"], errors="coerce").dropna()
                if not valid_ids.empty:
                    return int(valid_ids.max()) + 1
        except Exception:
            pass
    return 1


def clean_address(text):
    """Standardizes address text for TF-IDF vectorization."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return " ".join(cleaned.split())


def check_scanned_against_saved(scanned_address, file_path=CSV_FILE, threshold=SIMILARITY_THRESHOLD):
    """
    Compares the newly scanned address against existing entries in the CSV.
    Returns: (all_results, top_match_dict, is_matched)
    """
    cleaned_query = clean_address(scanned_address)
    if not os.path.exists(file_path) or not cleaned_query:
        return [], None, False

    try:
        df = pd.read_csv(file_path)
        if df.empty or "Address" not in df.columns:
            return [], None, False

        valid_rows = df[df["Address"].fillna("").str.strip().ne("") & (df["Address"] != "N/A")].copy()
        if valid_rows.empty:
            return [], None, False

        all_saved = [clean_address(a) for a in valid_rows["Address"].tolist()]
        corpus = all_saved + [cleaned_query]

        # Use character/word n-gram TF-IDF to handle minor OCR misspellings
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(corpus)

        similarity_vector = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        comparison_results = []
        for idx, row in valid_rows.reset_index(drop=True).iterrows():
            score_pct = round(float(similarity_vector[idx]) * 100, 2)
            comparison_results.append({
                "Serial_No": row.get("Serial_No", "Unknown"),
                "Saved_Address": str(row.get("Address", "")),
                "Similarity_Pct": score_pct,
                "Is_Match": score_pct >= (threshold * 100)
            })

        comparison_results.sort(key=lambda x: x["Similarity_Pct"], reverse=True)
        top_match = comparison_results[0] if comparison_results else None
        is_matched = top_match["Is_Match"] if top_match else False

        return comparison_results, top_match, is_matched

    except Exception as e:
        print(f"[!] Database read error: {e}")
        return [], None, False


def extract_address_only(raw_text):
    """Filters lines by common address tokens to isolate the address."""
    lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 3]

    address_keywords = [
        "stop", "bus", "road", "rd", "street", "st", "lane", "ln", "nagar", "plot",
        "opp", "opposite", "near", "nr", "bldg", "building", "flat", "floor", "dist",
        "district", "sector", "phase", "block", "pune", "mumbai", "delhi", "bangalore",
        "kolhapur", "goa", "bicholim", "chennai", "hyderabad", "avenue", "ave", "chowk"
    ]

    matched_lines = [
        line for line in lines
        if any(k in line.lower() for k in address_keywords)
        and "@" not in line
        and not line.lower().startswith("www.")
        and not line.lower().startswith("http")
    ]

    detected = " ".join(matched_lines) if matched_lines else raw_text.replace("\n", " ")
    return re.sub(r"\s+", " ", detected).strip()


def draw_overlay(image, is_matched, top_match, scanned_addr, total_checked):
    """Draws visual HUD banner indicating Match/No-Match status on the frame."""
    annotated = image.copy()
    h, w, _ = annotated.shape

    # Green for match, Orange-Red for no match
    badge_color = (0, 200, 0) if is_matched else (0, 70, 255)

    if is_matched:
        header = f"MATCHED: {top_match['Similarity_Pct']}% (Serial #{top_match['Serial_No']})"
        line1 = f"DB Addr: {top_match['Saved_Address'][:60]}..."
    else:
        best_pct = f"{top_match['Similarity_Pct']}%" if top_match else "0.0%"
        best_id = f"Serial #{top_match['Serial_No']}" if top_match else "None"
        header = f"NOT MATCHED (Top: {best_pct} with {best_id})"
        line1 = f"Checked {total_checked} records -> Added to DB as new entry"

    line2 = f"Scanned: {scanned_addr[:65]}"

    # Top Status Box
    cv2.rectangle(annotated, (15, 15), (w - 15, 75), (25, 25, 25), -1)
    cv2.rectangle(annotated, (15, 15), (w - 15, 75), badge_color, 2)
    cv2.putText(annotated, header, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, badge_color, 2)

    # Bottom Metadata Box
    cv2.rectangle(annotated, (15, h - 85), (w - 15, h - 15), (25, 25, 25), -1)
    cv2.rectangle(annotated, (15, h - 85), (w - 15, h - 15), badge_color, 1)
    cv2.putText(annotated, line1, (25, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(annotated, line2, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    return annotated


def append_address_to_csv(serial_no, address, status="NEW", file_path=CSV_FILE):
    """Appends unique verified addresses to CSV storage."""
    record = {"Serial_No": serial_no, "Address": address, "Status": status}
    df_new = pd.DataFrame([record])[CSV_COLUMNS]

    try:
        header_needed = not os.path.exists(file_path)
        df_new.to_csv(file_path, mode="a", header=header_needed, index=False)
        print(f"[✓] Successfully registered Serial #{serial_no} to {file_path}")
    except PermissionError:
        backup = f"scanned_addresses_backup_{int(time.time())}.csv"
        df_new.to_csv(backup, mode="w", header=True, index=False)
        print(f"[!] Target CSV locked. Saved to backup: {backup}")


# --- Camera Initialization ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Webcam not detected or accessible.")

current_serial = get_next_serial_number()

print("=" * 65)
print("  OCR CARD ADDRESS VERIFIER & DEDUPLICATOR")
print(f"  Next Serial Index : #{current_serial}")
print("  [SPACE] or [ENTER] : Capture Card & Check Address")
print("  [Q]               : Exit")
print("=" * 65)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_preview = frame.copy()
    cv2.putText(
        display_preview,
        f"Next Serial: #{current_serial} | Press ENTER to Scan",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )
    cv2.imshow("Card Scanner", display_preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (13, 32):  # ENTER or SPACE
        print(f"\n[*] Capturing {BURST_FRAME_COUNT} frames for noise reduction...")
        color_frames = []
        gray_frames = []

        for i in range(1, BURST_FRAME_COUNT + 1):
            ret_burst, b_frame = cap.read()
            if not ret_burst:
                continue

            color_frames.append(b_frame.astype(np.float32))
            gray_frames.append(cv2.cvtColor(b_frame, cv2.COLOR_BGR2GRAY).astype(np.float32))

            hud = b_frame.copy()
            cv2.putText(
                hud,
                f"Capturing: {i}/{BURST_FRAME_COUNT}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            cv2.imshow("Card Scanner", hud)
            cv2.waitKey(1)

        if not gray_frames:
            continue

        # Temporal frame averaging
        avg_color = np.mean(color_frames, axis=0).astype(np.uint8)
        avg_gray = np.mean(gray_frames, axis=0).astype(np.uint8)

        # Image preprocessing
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(avg_gray, -1, kernel)
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # OCR extraction
        raw_text = pytesseract.image_to_string(thresh, config="--psm 11").strip()
        if not raw_text:
            raw_text = pytesseract.image_to_string(sharpened).strip()

        scanned_address = extract_address_only(raw_text)

        if not scanned_address or len(scanned_address) < 5:
            print("[!] Could not detect a valid address from the frame. Please align and scan again.")
            continue

        print("\n" + "-" * 60)
        print(f"Scanned Address : {scanned_address}")
        print("-" * 60)

        # Check against database
        all_results, top_match, is_matched = check_scanned_against_saved(scanned_address)

        if all_results:
            print("Top Comparisons:")
            for item in all_results[:3]:
                status_flag = "[MATCH]" if item["Is_Match"] else "[NO MATCH]"
                print(f" -> Serial #{item['Serial_No']}: {item['Similarity_Pct']}% {status_flag} | {item['Saved_Address'][:50]}")

        # Render display overlay
        annotated_card = draw_overlay(
            avg_color,
            is_matched,
            top_match,
            scanned_address,
            len(all_results)
        )

        # Handle Match / Not Matched
        if is_matched:
            print("\n" + "=" * 60)
            print("STATUS: MATCHED")
            print(f"Matched With     : Serial #{top_match['Serial_No']} ({top_match['Similarity_Pct']}%)")
            print(f"Existing Record  : {top_match['Saved_Address']}")
            print("Action           : Discarded as duplicate (Not added to CSV).")
            print("=" * 60 + "\n")
        else:
            pct_disp = f"{top_match['Similarity_Pct']}%" if top_match else "0.0%"
            print("\n" + "=" * 60)
            print(f"STATUS: NOT MATCHED (Highest was {pct_disp})")
            print(f"Action           : Added to CSV as Serial #{current_serial}")
            print("=" * 60 + "\n")

            append_address_to_csv(current_serial, scanned_address, status="NEW")
            current_serial += 1

        # Save annotated image copy for audit
        img_out = os.path.join(IMAGE_DIR, f"scan_{int(time.time())}.jpg")
        cv2.imwrite(img_out, annotated_card)

        cv2.imshow("Address Comparison Result", annotated_card)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()