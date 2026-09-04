import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import pytesseract
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configure Tesseract Path ---
tesseract_cmd_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(tesseract_cmd_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path

CSV_FILE = "scanned_addresses.csv"
IMAGE_DIR = "scanned_images"
TOTAL_BURST_FRAMES = 100

# Strict minimal schema: only serial, address, and status
ADDRESS_CSV_COLUMNS = ["Serial_No", "Address", "Status"]

os.makedirs(IMAGE_DIR, exist_ok=True)


def get_next_serial_number(file_path=CSV_FILE):
    """Determines the next consecutive numeric Serial Number."""
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
    """Normalizes address string for TF-IDF comparison."""
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", str(text).lower())
    return " ".join(cleaned.split())


def check_scanned_against_all_saved_addresses(scanned_address, file_path=CSV_FILE, threshold=0.45):
    """Checks the scanned card address against all saved addresses in the CSV.
    Returns: (all_results_list, top_match_dict, is_matched)
    """
    if not os.path.exists(file_path) or not scanned_address.strip() or scanned_address == "N/A":
        return [], None, False

    try:
        df = pd.read_csv(file_path)
        if df.empty or "Address" not in df.columns:
            return [], None, False

        valid_rows = df[df["Address"].fillna("").str.strip().ne("") & (df["Address"] != "N/A")].copy()
        if valid_rows.empty:
            return [], None, False

        all_saved_addresses = [clean_address(a) for a in valid_rows["Address"].tolist()]
        cleaned_query = clean_address(scanned_address)

        if not cleaned_query.strip():
            return [], None, False

        # Compute TF-IDF matrix
        corpus = all_saved_addresses + [cleaned_query]
        vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Compute Cosine similarity
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
        print(f"Error reading database: {e}")
        return [], None, False


def extract_address_only(raw_text):
    """Extracts address lines from OCR text."""
    lines = [line.strip() for line in raw_text.splitlines() if len(line.strip()) > 2]

    address_keywords = [
        "stop", "bus", "road", "rd", "street", "st", "lane", "nagar", "plot",
        "opp", "near", "bldg", "flat", "floor", "dist", "kolhapur", "goa",
        "pune", "mumbai", "bichalim", "delhi", "bangalore", "chennai", "sector", "phase"
    ]
    
    address_lines = [
        l for l in lines
        if any(k in l.lower() for k in address_keywords) and "@" not in l and not l.lower().startswith("www.")
    ]
    
    detected_address = " ".join(address_lines) if address_lines else "N/A"
    return re.sub(r"\s+", " ", detected_address).strip()


def draw_overlay(image, is_matched, top_match, scanned_addr, total_checked):
    """Renders real-time match status and percentage overlay onto the frame."""
    annotated = image.copy()
    h, w, _ = annotated.shape

    badge_color = (0, 220, 0) if is_matched else (0, 70, 255)

    if is_matched:
        header = f"MATCHED: {top_match['Similarity_Pct']}% (Matches Serial #{top_match['Serial_No']})"
        line1 = f"DB Addr: {top_match['Saved_Address'][:60]}..."
    else:
        best_pct = f"{top_match['Similarity_Pct']}%" if top_match else "0.0%"
        best_id = f"Serial #{top_match['Serial_No']}" if top_match else "None"
        header = f"NOT MATCHED (Highest: {best_pct} with {best_id})"
        line1 = f"Checked {total_checked} entries -> Saved as new record"

    line2 = f"Scanned: {scanned_addr[:65]}"

    # Top Status Banner
    cv2.rectangle(annotated, (15, 15), (w - 15, 75), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, 15), (w - 15, 75), badge_color, 3)
    cv2.putText(annotated, header, (30, 55), cv2.FONT_HERSHEY_DUPLEX, 0.75, badge_color, 2)

    # Bottom Details Banner
    cv2.rectangle(annotated, (15, h - 85), (w - 15, h - 15), (20, 20, 20), -1)
    cv2.rectangle(annotated, (15, h - 85), (w - 15, h - 15), badge_color, 2)
    cv2.putText(annotated, line1, (25, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(annotated, line2, (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

    return annotated


def append_address_to_csv(serial_no, address, status="NEW", file_path=CSV_FILE):
    """Appends only Serial_No, Address, and Status to the CSV file."""
    record = {
        "Serial_No": serial_no,
        "Address": address,
        "Status": status
    }
    df_new = pd.DataFrame([record])[ADDRESS_CSV_COLUMNS]
    try:
        if os.path.exists(file_path):
            df_new.to_csv(file_path, mode="a", header=False, index=False)
        else:
            df_new.to_csv(file_path, mode="w", header=True, index=False)
        print(f"[✓] Saved new address to {file_path}")
    except PermissionError:
        backup = f"scanned_addresses_backup_{int(time.time())}.csv"
        df_new.to_csv(backup, mode="w", header=True, index=False)
        print(f"[!] File locked. Saved to backup: {backup}")


# --- Camera Loop ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

current_serial = get_next_serial_number()

print("=" * 70)
print(" ADDRESS MATCHER & DEDUPLICATOR")
print(f" Next Available Serial No: #{current_serial}")
print(" [ENTER] / [SPACE] -> Scan and Compare")
print(" [Q]               -> Quit")
print("=" * 70)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preview = frame.copy()
    cv2.putText(
        preview,
        f"Next Serial: #{current_serial} | Press ENTER to Scan",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.imshow("Address Scanner", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (13, 32):  # ENTER or SPACE
        print(f"\n[*] Scanning {TOTAL_BURST_FRAMES} frames...")
        color_frames = []
        gray_frames = []

        for i in range(1, TOTAL_BURST_FRAMES + 1):
            ret_burst, frame_burst = cap.read()
            if not ret_burst:
                continue

            color_frames.append(frame_burst.astype(np.float32))
            gray_frames.append(cv2.cvtColor(frame_burst, cv2.COLOR_BGR2GRAY).astype(np.float32))

            hud = frame_burst.copy()
            cv2.putText(
                hud,
                f"Scanning: {i}/{TOTAL_BURST_FRAMES}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.imshow("Address Scanner", hud)
            cv2.waitKey(1)

        if not gray_frames:
            continue

        # Noise reduction & OCR
        final_color_card = np.mean(color_frames, axis=0).astype(np.uint8)
        avg_gray = np.mean(gray_frames, axis=0).astype(np.uint8)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(avg_gray, -1, kernel)
        _, thresholded = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        raw_text = pytesseract.image_to_string(thresholded, config="--psm 11").strip()
        if not raw_text:
            raw_text = pytesseract.image_to_string(sharpened).strip()

        scanned_address = extract_address_only(raw_text)

        print("\n" + "-" * 60)
        print(f"Scanned Address: {scanned_address}")
        print("-" * 60)

        # Match against saved addresses
        all_results, top_match, is_matched = check_scanned_against_all_saved_addresses(scanned_address)

        # Print all comparison percentages to terminal
        if all_results:
            print("Comparison Results:")
            for item in all_results:
                flag = "[MATCHED]" if item["Is_Match"] else "[NO MATCH]"
                print(f" -> Serial #{item['Serial_No']}: {item['Similarity_Pct']}% match {flag}")
                print(f"    Saved: {item['Saved_Address']}")
        else:
            print("[i] Database empty. No prior addresses to compare against.")

        # Overlay results on frame
        stamped_image = draw_overlay(
            final_color_card,
            is_matched,
            top_match,
            scanned_address,
            len(all_results)
        )

        # Conditional action
        if is_matched:
            print("\n" + "=" * 60)
            print(f"RESULT: MATCHED")
            print(f"Match Percentage : {top_match['Similarity_Pct']}%")
            print(f"Matched With     : Serial #{top_match['Serial_No']}")
            print(f"DB Address       : {top_match['Saved_Address']}")
            print(f"Action           : NOT saved to CSV (Duplicate).")
            print("=" * 60 + "\n")
        else:
            pct_display = f"{top_match['Similarity_Pct']}%" if top_match else "0.0%"
            print("\n" + "=" * 60)
            print(f"RESULT: NOT MATCHED (Top similarity was {pct_display})")
            print(f"Action: Saved to CSV as Serial #{current_serial}")
            print("=" * 60 + "\n")

            append_address_to_csv(current_serial, scanned_address, status="NEW")
            current_serial += 1

        cv2.imshow("Address Comparison Result", stamped_image)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()