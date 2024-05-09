import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def Region_mask(mask,center_of_mass,height,width):
    mask_region = np.zeros_like(mask)
    if not np.isnan(center_of_mass[0]) and not np.isnan(center_of_mass[1]):
        center_of_mass = (round(center_of_mass[0]), round(center_of_mass[1]))
        # Calculate the rectangle boundaries
        top = max(center_of_mass[1] - height // 2, 0)  # Ensure top is not less than 0
        bottom = min(center_of_mass[1] + height // 2, mask.shape[0])
        left = max(center_of_mass[0] - width // 2, 0)  # Ensure left is not less than 0
        right = min(center_of_mass[0] + width // 2, mask.shape[1])  # Ensure right does not exceed mask width

        # Replace the region within the bounds with the corresponding values from the original mask
        mask_region[top:bottom, left:right] = mask[top:bottom, left:right]
    # iris
    return mask_region

def filter_player(frame, background):
    """Processes the video frames to extract the foreground where the player might be located.
    It uses techniques like Gaussian blur, median blur,
     and thresholding to filter out the player from the background."""
    # Compute the absolute difference of the current frame and background
    diff = cv2.absdiff(frame, background)
    # Convert the difference image to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter to smooth the image
    diff_smoothed = cv2.GaussianBlur(diff_gray, (15, 15), 20)

    # Apply Median filter to further reduce noise
    diff_smoothed1 = cv2.medianBlur(diff_smoothed, 9)
    diff_smoothed = cv2.medianBlur(diff_smoothed1, 9)
    diff_smoothed = cv2.medianBlur(diff_smoothed, 9)

    # Threshold the diff image so that we get the foreground
    _, thresh = cv2.threshold(diff_smoothed, 25, 255, cv2.THRESH_BINARY)
    _, thresh_for_color = cv2.threshold(diff_smoothed1, 25, 255, cv2.THRESH_BINARY)

    return thresh, thresh_for_color
def get_player_position(mask,outlier_std_threshold=5,only_center = 0):
    """
    :param mask: binary mask
    :return: (center_x, center_y)
    """
    # Find indices where we have mass
    mass_h, mass_w = np.where(mask == 255)

    # x,y are the center of x indices and y indices of mass pixels
    center_of_mass = (np.average(mass_w), np.average(mass_h))
    # if only_center == 1:
    #     return center_of_mass, 10, 10, 0
    # if len(mass_w) < 10 or len(mass_h) < 10:
    #     #center_of_mass = (mask.shape[0]//2,mask.shape[1]//2)
    #     return center_of_mass, 10, 10, 0

    # Calculate distances of each pixel from the center of mass
    distances = np.sqrt((mass_h - center_of_mass[1]) ** 2 + (mass_w - center_of_mass[0]) ** 2)

    # Filter out outliers based on the standard deviation threshold
    std_dev = np.std(distances)
    main_mass_indices = np.where(distances <= outlier_std_threshold * std_dev)

    # Use only main mass indices to calculate width and height
    main_mass_w = mass_w[main_mass_indices]
    main_mass_h = mass_h[main_mass_indices]

    # Calculate percentage of mask pixels being equal to 1
    total_pixels = mask.shape[0] * mask.shape[1]
    ones_count = np.count_nonzero(mask)
    percentage = (ones_count / total_pixels) * 100

    # Cancelled for now
    # if len(main_mass_w) < 50 or len(main_mass_h) < 50:
    #     width = 50
    #     height = 50
    #     return center_of_mass, width, height, percentage

    width = np.max(main_mass_w) - np.min(main_mass_w)
    height = np.max(main_mass_h) - np.min(main_mass_h)
    mask_region = Region_mask(mask, center_of_mass, height, width)

    # Find indices where we have mass
    mass_h, mass_w = np.where(mask_region == 255)
    # x,y are the center of x indices and y indices of mass pixels
    center_of_mass = (np.average(mass_w), np.average(mass_h))

    # Lower body region mask
    mask_upper_region = mask_region.copy()

    mask_upper_region[int(center_of_mass[0]):,:] = 0

    # # show the mask
    # plt.imshow(mask_upper_region)
    # plt.title("mask_upper_region")
    # plt.show()

    # Lower body region mask
    mask_lower_region = mask_region.copy()
    mask_lower_region[:int(center_of_mass[0]), :] = 0


    upper_mass_h, upper_mass_w = np.where(mask_upper_region == 255)
    center_of_upper_mass = np.average(upper_mass_w), np.average(upper_mass_h)

    lower_mass_h, lower_mass_w = np.where(mask_lower_region == 255)
    center_of_lower_mass = np.average(lower_mass_w), np.average(lower_mass_h)


    return center_of_mass, center_of_upper_mass, center_of_lower_mass





# load imges from the dataset
lean = cv2.imread("assets/lean_3.jpg")
bg = cv2.imread("assets/background.jpg")

# show the images
cv2.imshow("lean", lean)
cv2.imshow("bg", bg)

#convert the images to RGB
lean = cv2.cvtColor(lean, cv2.COLOR_BGR2RGB)
bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

mask, _ = filter_player(lean, bg)
def get_centers(img,bg):
    mask, _ = filter_player(img, bg)
    center, upper_center, lower_center = get_player_position(mask, 5)
    return center, upper_center, lower_center


center, upper_center,lower_center = get_centers(lean, bg)

def mark_centers(img, center, upper_center, lower_center):
    img1 = img.copy()
    img1 = cv2.circle(img, (int(center[0]), int(center[1])), 10, (255, 0, 0), -1)
    img1 = cv2.circle(img, (int(upper_center[0]), int(upper_center[1])), 10, (0, 255, 0), -1)
    img1 = cv2.circle(img, (int(lower_center[0]), int(lower_center[1])), 10, (0, 0, 255), -1)
    return img1
# mark the center of mass
mark_centers(lean, center, upper_center, lower_center)

#print(f'centers of mass: {center, upper_center,lower_center }')
print(f'lean center of mass: {upper_center[0]-lower_center[0]}')
plt.imshow(lean)
plt.title("lean")
plt.show()

plt.imshow(cv2.absdiff(lean, bg))
plt.title("bg")
plt.show()

plt.imshow(mask)
plt.title("mask")
plt.show()
#

DELTA = 13
def detect_lean(img,background):
    center, upper_center, lower_center = get_centers(img, background)
    lean = upper_center[0] - lower_center[0]
    if lean < -DELTA:
        return "left"
    if lean > DELTA:
        return "right"
    return "idle"


def add_salt_pepper_noise(image, salt_pepper_ratio=0.5, amount=0.01):
    """
    Add salt and pepper noise to an image.
    Args:
        image (numpy.array): The input image.
        salt_pepper_ratio (float): The proportion of salt vs. pepper noise.
        amount (float): The percentage of image pixels to be affected by noise.
    Returns:
        numpy.array: The noisy image.
    """
    # Create a copy of the image to avoid modifying the original
    noisy_image = np.copy(image)

    # Calculate the number of pixels to be affected by noise
    num_salt = np.ceil(amount * image.size * salt_pepper_ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - salt_pepper_ratio))

    # Add Salt noise (white pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Add Pepper noise (black pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

NOISE_TYPE = 'salt_and_pepper' # 'gaussian' or 'salt_and_pepper'
if __name__ == "__main__":
    bg = cv2.imread("assets/background.jpg")
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    # Path to the CSV file
    csv_file_path = 'labeled_images.csv'

    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    #delete the first row of the DataFrame
    df = df.drop(0)
    df['noisy_0'] = 'idle'

    #iterate through the rows of the DataFrame
    for index, row in df.iterrows():
        #load the image
        img = cv2.imread(row['path'])
        #convert the image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # create new column to store the lean

        for i in range(6):
            # add noise to the image of different magnitudes
            if NOISE_TYPE == 'gaussian':
                noise = np.random.normal(-i*2, i*2, lean.shape).astype(np.uint8)
                noisy = cv2.add(img, noise)
            elif NOISE_TYPE == 'salt_and_pepper':
                noisy = add_salt_pepper_noise(img, amount=i/100)

            # get the plots on the noisy image
            mask, _ = filter_player(noisy, bg)
            center, upper_center, lower_center = get_centers(noisy, bg)
            noisy1 = mark_centers(noisy, center, upper_center, lower_center)

            plt.imshow(noisy1)
            plt.title(f"noisy {i}")
            plt.show()

            plt.imshow(mask)
            plt.title(f"mask {i}")
            plt.show()

            # detect the lean
            df.at[index, f'noisy_{i}'] = detect_lean(noisy, bg)
            print(detect_lean(img, bg))

    # Save DataFrame to CSV
    df.to_csv(f'data {NOISE_TYPE} noise.csv', index=False)
    # display the first few rows of the DataFrame
    print(f"df after noise 0:\n {df}")