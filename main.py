import subprocess

def run_demo(transformation, feature_extraction, sequence_modeling, prediction, image_folder, saved_model):
    command = f"python demo.py --Transformation {transformation} --FeatureExtraction {feature_extraction} --SequenceModeling {sequence_modeling} --Prediction {prediction} --image_folder {image_folder} --saved_model {saved_model}"
    subprocess.run(command, shell=True)

# Example usage:
run_demo("TPS", "ResNet", "BiLSTM", "Attn", "cropped_images/", "TPS-ResNet-BiLSTM-Attn.pth")
#run_demo("None", "VGG", "None", "CTC", "cropped_images/", "None-VGG-None-CTC.pth")

image_path="img_89.jpg"
# Open merged_info.txt in write mode to clear its contents
with open('merged_info.txt', 'w') as merged_file:
    # Open both files for reading
    with open('box_coordinates.txt', 'r') as boxes_file, open('recognized_texts.txt', 'r') as texts_file:
        # Iterate over lines in both files simultaneously
        for box_line, text_line in zip(boxes_file, texts_file):
            # Split lines to extract image path, box coordinates, text, and score
            box_info = box_line.strip().split()
            text_info = text_line.strip().split()
            
            # Extract information
            box_coordinates = ' '.join(box_info[1:])
            text = text_info[1]
            score = text_info[2]
            
            # Write merged information to the new file
            merged_file.write(f"{box_coordinates} {text} {score}\n")

import cv2

# Read the text file
with open("merged_info.txt", "r") as file:
    lines = file.readlines()

# Load the image
image = cv2.imread("img_91.jpg")

# Dictionary to store bounding boxes grouped by first four letters of labels
grouped_boxes = {}

# Iterate through each line in the text file
for line in lines:
    # Split the line into components: x1, y1, x2, y2, label, score
    x1, y1, x2, y2, label, score = line.strip().split()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    score = float(score)

    # Check if the score is greater than 0.5
    if score > 0.24:
        # Get the first four letters of the label
        first_four_letters = label[:4]

        # Check if there is a label with the same first four letters
        if first_four_letters in grouped_boxes:
            # Compare the length of the labels
            if len(label) > len(grouped_boxes[first_four_letters][0][4]):
                # Replace the stored bounding boxes with the longer label
                grouped_boxes[first_four_letters] = [(x1, y1, x2, y2, label, score)]
            elif len(label) == len(grouped_boxes[first_four_letters][0][4]):
                # If the labels have the same length, append the bounding box
                grouped_boxes[first_four_letters].append((x1, y1, x2, y2, label, score))
        else:
            # If no label with the same first four letters, add a new entry
            grouped_boxes[first_four_letters] = [(x1, y1, x2, y2, label, score)]

# Iterate through grouped boxes and draw the bounding boxes on the image
for group in grouped_boxes.values():
    for box in group:
        x1, y1, x2, y2, label, score = box
        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label and score on the image
        cv2.putText(image, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the image
cv2.imshow("Bounding Boxes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image with bounding boxes drawn
cv2.imwrite("output_image.jpg", image)