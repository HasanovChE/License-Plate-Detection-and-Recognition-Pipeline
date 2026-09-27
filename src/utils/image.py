import cv2

def crop_with_padding(
    image,
    bbox,
    padding_ratio=0.15
):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    box_width = x2 - x1
    box_height = y2 - y1
    
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)
    
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    
    if x2 <= x1 or y2 <= y1:
        return None
        
    return image[y1:y2, x1:x2]