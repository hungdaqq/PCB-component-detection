import pandas as pd
import cv2
import os
import ast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define paths
labels_folder = "/home/hung/Downloads/pcb/ocr_annotation"
images_folder = "/home/hung/Downloads/pcb/pcb_image"
output_folder = "./test"

# Ensure output folder and orientation subfolders exist
os.makedirs(output_folder, exist_ok=True)
for orientation in [0, 90, 180, 270]:
    os.makedirs(os.path.join(output_folder, str(orientation)), exist_ok=True)

# Get a list of all CSV files in the labels folder
csv_files = [f for f in os.listdir(labels_folder) if f.endswith(".csv")]

logging.info(f"Found {len(csv_files)} CSV files.")

# Process each CSV file
for csv_file in csv_files:
    csv_file_path = os.path.join(labels_folder, csv_file)

    # Read the CSV file
    labels_df = pd.read_csv(csv_file_path)
    logging.info(f"Processing file: {csv_file_path}")

    # Iterate through each row in the DataFrame
    for index, row in labels_df.iterrows():
        image_filename = csv_file_path.replace("_ocr.csv", ".png")
        image_filename = os.path.basename(image_filename)
        coordinates_str = row["Vertices"]
        orientation = row["Orientation"]
        instance_id = row["Instance ID"]

        # Ensure the orientation is valid
        if orientation not in [0, 90, 180, 270]:
            logging.warning(
                f"Skipping invalid orientation {orientation} in file {csv_file}, row {index}"
            )
            continue

        # Parse the coordinates using ast.literal_eval for safety
        try:
            coordinates_list = ast.literal_eval(coordinates_str)
        except (ValueError, SyntaxError):
            logging.error(
                f"Skipping invalid coordinates in file {csv_file}, row {index}"
            )
            continue

        for i, coordinates in enumerate(coordinates_list):
            # Calculate the bounding box
            left, top = coordinates[0]
            right, bottom = coordinates[2]

            # Open the image using OpenCV
            image_path = os.path.join(images_folder, image_filename)
            try:
                img = cv2.imread(image_path)
                if img is None:
                    logging.error(f"Failed to load image {image_path}")
                    continue

                # Crop the image
                cropped_img = img[top:bottom, left:right]

                # Create unique output filenames
                base_filename, ext = os.path.splitext(image_filename)
                output_image_filename = f"{base_filename}_instance{instance_id}{ext}"
                output_image_path = os.path.join(
                    output_folder, str(orientation), output_image_filename
                )

                # Save the cropped image
                cv2.imwrite(output_image_path, cropped_img)
                logging.info(f"Saved cropped image to: {output_image_path}")

            except Exception as e:
                logging.error(f"Error processing image {image_filename}: {e}")

logging.info("Processing complete.")
