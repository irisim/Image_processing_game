
import cv2
def add_grid_scale(image):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Define the scale parameters
    scale_length = min(height, width) // 10  # Length of the scale in pixels (1/10th of the smaller dimension)
    scale_thickness = 2  # Thickness of the scale lines
    scale_color = (255, 0, 0)  # Color of the scale lines
    scale_text_color = (255, 255, 255)  # Color of the scale text
    scale_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for the scale text
    scale_font_size = 0.5  # Font size for the scale text

    # Draw the vertical axis scale
    cv2.line(image, (1, 1), (1, height - 1), scale_color, scale_thickness)
    cv2.line(image, (width // 4, 1), (width // 4, height - 1), scale_color, scale_thickness)
    cv2.line(image, (width // 2, 1), (width // 2, height - 1), scale_color, scale_thickness)
    cv2.line(image, (3 * width // 4, 1), (3 * width // 4, height - 1), scale_color, scale_thickness)
    cv2.putText(image, '0', (20, 20), scale_font, scale_font_size, scale_text_color, 1, cv2.LINE_AA)
    cv2.putText(image, str(height // 4), (20, height // 4 - 3), scale_font, scale_font_size, scale_text_color, 1,
                cv2.LINE_AA)
    cv2.putText(image, str(height // 2), (20, height // 2 - 3), scale_font, scale_font_size, scale_text_color, 1,
                cv2.LINE_AA)
    cv2.putText(image, str(height), (20, height - 5), scale_font, scale_font_size, scale_text_color, 1, cv2.LINE_AA)
    cv2.putText(image, str(5 * height // 8), (20, 5 * height // 8 - 3), scale_font, scale_font_size, scale_text_color,
                1, cv2.LINE_AA)
    cv2.putText(image, str(3 * height // 4), (20, 3 * height // 4 - 3), scale_font, scale_font_size, scale_text_color,
                1, cv2.LINE_AA)
    cv2.putText(image, str(7 * height // 8), (20, 7 * height // 8 - 3), scale_font, scale_font_size, scale_text_color,
                1, cv2.LINE_AA)

    # Draw the horizontal axis scale
    cv2.line(image, (1, 1), (width - 1, 1), scale_color, scale_thickness)
    cv2.line(image, (1, height // 4), (width - 1, height // 4), scale_color, scale_thickness)
    cv2.line(image, (1, height // 2), (width - 1, height // 2), scale_color, scale_thickness)
    cv2.line(image, (1, 5 * height // 8), (width - 1, 5 * height // 8), (0, 255, 0), scale_thickness)
    cv2.line(image, (1, 3 * height // 4), (width - 1, 3 * height // 4), scale_color, scale_thickness)
    cv2.line(image, (1, 7 * height // 8), (width - 1, 7 * height // 8), (0, 255, 0), scale_thickness)
    cv2.putText(image, str(width // 4), (width // 4 - 20, 20), scale_font, scale_font_size, scale_text_color, 1,
                cv2.LINE_AA)
    cv2.putText(image, str(width // 2), (width // 2 - 20, 20), scale_font, scale_font_size, scale_text_color, 1,
                cv2.LINE_AA)
    cv2.putText(image, str(3 * width // 4), (3 * width // 4 - 20, 20), scale_font, scale_font_size, scale_text_color, 1,
                cv2.LINE_AA)
    cv2.putText(image, str(width), (width - 30, 20), scale_font, scale_font_size, scale_text_color, 1, cv2.LINE_AA)

    return image
