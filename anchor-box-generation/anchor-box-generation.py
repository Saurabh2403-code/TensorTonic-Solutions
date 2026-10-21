def generate_anchors(feature_size: int, image_size: float, scales: list[float], aspect_ratios: list[float]) -> list[list[float]]:
    """
    Returns a list of [x1, y1, x2, y2] anchor boxes.
    """
    stride=image_size//feature_size
    x_coordinates=[(0.5+i)*stride for i in range(feature_size)]
    y_coordinates=x_coordinates
    boxes=[]
    for s in scales:
        for r in aspect_ratios:
            h=s/(r**0.5)
            w=s*(r**0.5)
            boxes.append((w,h))
    anchor_box=[]
    for j in x_coordinates:
        for i in y_coordinates:
            for w,h in boxes:
                anchor_box.append([i-w/2,j-h/2,i+w/2,j+h/2])
    return anchor_box
            
            
    
    