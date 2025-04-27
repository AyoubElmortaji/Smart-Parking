import cv2
import numpy as np
import os
from datetime import datetime

def calculate_parking_fee(hours):
    
    if hours <= 0:
        return 0
    if hours <= 1:
        return 2
    if hours >= 10:
        return 10
    return 2 + (hours - 1) * 1

def detect_parking_spots():
    # --- SETTINGS ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join('static', 'images', f'captured_{timestamp}.jpg')
    output_path = os.path.join('static', 'images', f'processed_{timestamp}.jpg')
    url = "http://192.168.86.33:8080/video"
    
    
    parking_times = {}  # Format: {spot_id: check_in_time}

    # --- CAPTURE ---
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return {"error": "Impossible d'accéder au flux IP Webcam."}
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
    else:
        return {"error": "Échec de la capture d'image."}
    cap.release()

    # --- LOAD IMAGE ---
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to load captured image."}
    
    output = img.copy()
    H, W, _ = img.shape

    # --- CONFIG ---
    blocks_top = 2    # 2 blocks in top row
    blocks_bottom = 2 # 2 blocks in bottom row
    slots_per_block = 2  # Each block has 2 slots
    slot_w, slot_h = 240, 300
    slot_spacing = 60
    vertical_spacing = 250
    left_offset = 200  # Increased left shift (was 150)
    block_spacing = 80  # Space between blocks

    free_spots = []
    occupied_spots = []
    parking_spots = []
    slot_id = 1

    # --- Draw Blocks and Slots ---
    for row in range(2):
        current_blocks = blocks_top if row == 0 else blocks_bottom
        for block in range(current_blocks):
            # Calculate block position (strong left shift)
            total_block_width = slots_per_block * slot_w + (slots_per_block - 1) * slot_spacing
            block_x = left_offset + block * (total_block_width + block_spacing)
            block_y = (H - (2 * slot_h + vertical_spacing)) // 2 + row * (slot_h + vertical_spacing)

            for slot in range(slots_per_block):
                x = block_x + slot * (slot_w + slot_spacing)
                y = block_y

                parking_spots.append((slot_id, x, y, slot_w, slot_h, row, block))

                roi = img[y:y + slot_h, x:x + slot_w]
                if roi.size == 0:
                    continue

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                white_pixels = cv2.countNonZero(thresh)
                total_pixels = slot_w * slot_h

                if white_pixels / total_pixels < 0.9:
                    status = "Occupied"
                    color = (0, 0, 255)  # Red for occupied
                    occupied_spots.append(slot_id)
                    
                    # Record parking time if this spot was just occupied
                    if slot_id not in parking_times:
                        parking_times[slot_id] = datetime.now()
                else:
                    status = "Free"
                    color = (0, 255, 0)  # Green for free
                    free_spots.append(slot_id)
                    
                    # Clear parking time if spot is now free
                    if slot_id in parking_times:
                        del parking_times[slot_id]

                cv2.rectangle(output, (x, y), (x + slot_w, y + slot_h), color, 2)
                cv2.putText(output, f"#{slot_id}", (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                slot_id += 1

            # Draw block outline
            block_width = slots_per_block * slot_w + (slots_per_block - 1) * slot_spacing
            cv2.rectangle(output,
                         (block_x - 10, block_y - 10),
                         (block_x + block_width + 10, block_y + slot_h + 10),
                         (255, 255, 0), 3)

            block_name = chr(65 + block + (row * blocks_top))
            cv2.putText(output, f"Block {block_name}", (block_x + 5, block_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- Suggest Best Free Spot ---
    def distance_to_entrance(x, y, w, h):
        center_x = x + w // 2
        center_y = y + h // 2
        entrance_x = W // 2
        entrance_y = H
        return np.hypot(center_x - entrance_x, center_y - entrance_y)

    best_spot_info = None
    if free_spots:
        free_slot_data = [
            (slot_id, x, y, w, h) for (slot_id, x, y, w, h, _, _) in parking_spots if slot_id in free_spots
        ]
        best_spot = min(free_slot_data, key=lambda s: distance_to_entrance(s[1], s[2], s[3], s[4]))

        slot_id, x, y, w, h = best_spot
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(output, "Best Spot", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        best_spot_info = f"#{slot_id}"

    # --- Calculate Parking Fees ---
    parking_fees = {}
    current_time = datetime.now()
    
    for spot_id, check_in_time in parking_times.items():
        hours_parked = (current_time - check_in_time).total_seconds() / 3600
        fee = calculate_parking_fee(hours_parked)
        parking_fees[spot_id] = {
            "check_in": check_in_time.strftime("%Y-%m-%d %H:%M:%S"),
            "hours": round(hours_parked, 2),
            "fee": fee
        }

    # --- SAVE ---
    cv2.imwrite(output_path, output)
    
    return {
        "status": "success",
        "original_image": image_path,
        "processed_image": output_path,
        "free_spots": free_spots,
        "occupied_spots": occupied_spots,
        "best_spot": best_spot_info,
        "parking_fees": parking_fees,
        "timestamp": timestamp
    }