import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# Open the video file
video_capture = cv2.VideoCapture('Panto2024.mp4')

# Get frame dimensions and frame rate for the video writer
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.avi', fourcc, frame_rate, (frame_width, frame_height))



def calculate_intersection(line1, line2):
    # Unpack line coordinates
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate slopes (m1, m2) and y-intercepts (c1, c2) of the lines
    m1 = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')  # Avoid division by zero
    c1 = y1 - m1 * x1

    m2 = (y4 - y3) / (x4 - x3) if x4 != x3 else float('inf')
    c2 = y3 - m2 * x3

    # Check if lines are parallel (slopes are equal)
    if m1 == m2:
        return None  # No intersection (parallel lines)

    # Calculate intersection point
    x_intersect = (c2 - c1) / (m1 - m2)
    y_intersect = m1 * x_intersect + c1

    return (x_intersect, y_intersect)

# Initialize a list to store intersection points
intersection_points = []
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if no frame is read
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 30, 150)
    
    # Apply Probabilistic Hough Transform for line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=150, maxLineGap=30)

    lowest_line = None
    left_line = None
    
    # Draw detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Check if the angle is within the specified range for lowest line
            if 0 <= angle <= 10 and y1 > 200 and y1 < 480: # Ensure that it's not the noise line
                if lowest_line is None or y1 < lowest_line[0][1]:  # Update lowest line
                    lowest_line = line
            
            # Check if the angle is within the specified range for left line
            if 60 <= angle <= 120 and y1 < 300: # Ensure that it's not the noise line
                if left_line is None or x1 < left_line[0][0]:  # Update left line
                    left_line = line
        
        # Draw the lowest line
        if lowest_line is not None:
            x1, y1, x2, y2 = lowest_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        # Draw the left line
        if left_line is not None:
            x1, y1, x2, y2 = left_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Calculate intersection point
            if lowest_line is not None and left_line is not None:
                x_intersect, y_intersect = calculate_intersection(lowest_line[0], left_line[0])
                cv2.circle(frame, (int(x_intersect), int(y_intersect)), 10, (0, 255, 0), -1)  # Draw a red dot at the intersection point
                intersection_points.append((x_intersect, y_intersect)) 
                print(x_intersect, y_intersect)
        out.write(frame)

    # Display the result
    cv2.imshow('Line Detection Result', frame)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
out.release()
cv2.destroyAllWindows()

# Convert the stored intersection points to a Pandas DataFrame
df = pd.DataFrame(intersection_points, columns=['X-coordinate', 'Y-coordinate'])

# Save the DataFrame to a CSV file
df.to_csv('intersection_points.csv', index=False)
