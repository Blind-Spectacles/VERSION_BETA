import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
import time

def initialize_sam(checkpoint_path, model_type="vit_b", device="cuda"):
    """
    Initialize the SAM model
    Args:
        checkpoint_path: Path to the SAM checkpoint file
        model_type: Type of SAM model (vit_b for base model)
        device: Device to run the model on (cuda or cpu)
    Returns:
        predictor: SAM predictor object
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)

def process_frame(frame, predictor, points_per_side=12):
    """
    Process a single frame with SAM
    Args:
        frame: Input frame from webcam
        predictor: SAM predictor object
        points_per_side: Number of points to sample (reduced for real-time performance)
    Returns:
        overlay: Frame with mask overlay
    """
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Set image in predictor
    predictor.set_image(frame_rgb)
    
    # Generate grid of points
    h, w = frame.shape[:2]
    rows = np.linspace(0, h, points_per_side, endpoint=False)
    cols = np.linspace(0, w, points_per_side, endpoint=False)
    points_grid = np.stack(np.meshgrid(cols, rows), axis=-1).reshape(-1, 2)
    
    # Generate masks
    masks, scores, _ = predictor.predict(
        point_coords=points_grid,
        point_labels=np.ones(len(points_grid)),
        multimask_output=False,
        mask_input=None
    )
    
    # Create overlay
    overlay = frame.copy()
    for mask, score in zip(masks, scores):
        if score >= 0.90:  # Confidence threshold
            color = np.random.randint(0, 255, (3,), dtype=np.uint8)
            mask_colored = (mask[:, :, None] * color).astype(np.uint8)
            overlay = cv2.addWeighted(overlay, 1, mask_colored, 0.5, 0)
    
    return overlay

def main():
    checkpoint_path = "sam_checkpoints/sam_vit_b.pth"  # Ensure this path is correct
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        predictor = initialize_sam(checkpoint_path, model_type="vit_b", device=device)
    except Exception as e:
        print(f"Error initializing SAM: {e}")
        cap.release()
        return
    
    print("Press 'q' to quit")
    
    fps_counter = 0
    fps_start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        start_time = time.time()
        result_frame = process_frame(frame, predictor)
        fps_counter += 1
        
        if time.time() - fps_start_time > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("SAM Webcam Detection", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
