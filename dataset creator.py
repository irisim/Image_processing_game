import os
import pandas as pd
from matplotlib import pyplot as plt

# Folder containing images
image_folder = 'assets'

# List to keep track of image paths and labels
data = []

# Iterate through each image in the folder
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):  # check for image files
        image_path = os.path.join(image_folder, image_name)

        # Display the image
        img = plt.imread(image_path)
        plt.imshow(img)
        plt.axis('off')  # turn off axis
        plt.show()

        # Input label
        label = input("Enter the label for the above image: ")

        # Store the image path and label in the list
        data.append({'path': image_path, 'label': label})

# Convert list to DataFrame
df = pd.DataFrame(data)

# Optional: Save DataFrame to CSV
df.to_csv('labeled_images.csv', index=False)

print("Labeling complete. Data saved to 'labeled_images.csv'.")