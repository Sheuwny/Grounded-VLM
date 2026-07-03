def quantize_box(bbox, width, height, num_bins=1000):
    # bbox format: [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox
    
    # Normalize and scale
    qx1 = int(x1 / width * num_bins)
    qy1 = int(y1 / height * num_bins)
    qx2 = int(x2 / width * num_bins)
    qy2 = int(y2 / height * num_bins)
    
    return [f"<loc{qx1}>", f"<loc{qy1}>", f"<loc{qx2}>", f"<loc{qy2}>"]

# Example test
print(quantize_box([10, 10, 50, 50], 100, 100))
