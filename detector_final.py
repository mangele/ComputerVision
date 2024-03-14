import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.ndimage import gaussian_filter1d
# Open the video file
video_capture = cv2.VideoCapture('Panto2024.mp4')

# Get frame dimensions and frame rate for the video writer
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video_old.avi', fourcc, frame_rate, (frame_width, frame_height))


# Initialize Matplotlib plot for real-time updating
plt.ion()
fig, ax = plt.subplots(figsize=(8,10))
xdata, ydata = [], []
ln, = plt.plot(xdata, ydata, 'ro-', markersize=2)  # 'ro-' for red dots with lines
plt.xlim(720, 1100)   # Set the x-axis limits to match the frame width
plt.ylim(480, 900)  # Set the y-axis limits to match the frame height
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Real-Time Intersection Point Tracking')


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


#############
from collections import deque
# Define your Gaussian kernel size



# Initialize a circular buffer with a fixed size of 10
buffer_size = 15
intersection_points = deque(maxlen=buffer_size)

gradient_list = deque(maxlen=12)
# Initialize a list to store intersection points
#intersection_points = []
record = []
threshold = 8
previous_stabilized_point = (0, 0)  
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()
    if not ret:
        break  # Break the loop if no frame is read
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    #blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 30, 150)
    
    # Apply Probabilistic Hough Transform for line detection
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=150, maxLineGap=30)

    horizontal_line = None
    vertical_line = None

    empty_point = None
    # Draw detected lines on the original frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Check if the angle is within the specified range for lowest line
            if 0 <= angle <= 10 and y1 > 200 and y1 < 480: # Ensure that it's not the noise line
                if horizontal_line is None or y1 < horizontal_line[0][1]:  # Update lowest line
                    horizontal_line = line
            
            # Check if the angle is within the specified range for left line
            if 60 <= angle <= 120 and y1 < 300: # Ensure that it's not the noise line
                if vertical_line is None or x1 < vertical_line[0][0]:  # Update left line
                    vertical_line = line
        
        # Draw the lowest line
        if horizontal_line is not None:
            x1, y1, x2, y2 = horizontal_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
        # Draw the left line
        if vertical_line is not None:
            x1, y1, x2, y2 = vertical_line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # Inside your loop, after calculating the intersection point:
            if horizontal_line is not None and vertical_line is not None:
                # Calculate intersection point
                x_intersect, y_intersect = calculate_intersection(horizontal_line[0], vertical_line[0])
                
                # Add the new point to the circular buffer
                intersection_points.append((x_intersect, y_intersect))


    if len(intersection_points) == buffer_size:
        # Calculate the median for x and y coordinates
        x_vals, y_vals = zip(*intersection_points)
        x_median = np.median(x_vals)
        y_median = np.median(y_vals)

        # If the change is within the threshold, update the stabilized point
        stabilized_point = (int(x_median), int(y_median))
        cv2.circle(frame, stabilized_point, 15, (0, 0, 255), -1)

       # Update plot data
        xdata.append(x_median)
        ydata.append(frame_height - y_median)
        ln.set_data(xdata, ydata)
        ax.relim()        # Recalculate limits
        ax.autoscale_view(True,True,True)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Draw a scaled arrow from the last point to the current point if there is a previous point
        if len(record) > 0:
            start_point = record[-1]
            end_point = stabilized_point

            # Calculate a scaled end point for a larger arrow
            scale_factor = 10  # Increase this factor to make the arrow larger
            scaled_end_x = start_point[0] + scale_factor * (end_point[0] - start_point[0])
            scaled_end_y = start_point[1] + scale_factor * (end_point[1] - start_point[1])
            scaled_end_point = (scaled_end_x, scaled_end_y)

            color = (0, 255, 0)  # Red color for better visibility
            thickness = 4  # Thickness of the arrow

            cv2.arrowedLine(frame, start_point, scaled_end_point, color, thickness) #tipLength=tip_length)


        # Optional: Record the stabilized coordinates
        record.append(stabilized_point)

            # Calculate intersection point
           # if horizontal_line is not None and vertical_line is not None:
           #     x_intersect, y_intersect = calculate_intersection(horizontal_line[0], vertical_line[0])
           #     cv2.circle(frame, (int(x_intersect), int(y_intersect)), 10, (0, 255, 0), -1)  # Draw a red dot at the intersection point
           #     intersection_points.append((x_intersect, y_intersect)) 
           #     print(x_intersect, y_intersect)


        #out.write(frame)

    # Display the result
    cv2.imshow('Line Detection Result', frame)
    
    # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#plt.savefig('intersection_point_tracking.png')

# Release the video capture object and close all windows
video_capture.release()
#out.release()
cv2.destroyAllWindows()

# Convert the stored intersection points to a Pandas DataFrame
#df = pd.DataFrame(intersection_points, columns=['X-coordinate', 'Y-coordinate'])
df = pd.DataFrame(record, columns=['X-coordinate', 'Y-coordinate'])



x = np.array(df['X-coordinate'])
y = np.array(df['Y-coordinate'])

plt.figure(figsize=(5,10))
plt.scatter(x, y, s=5)
plt.scatter(x[0], y[0], s=20, color='red')
plt.scatter(x[-1], y[-1], s=20, color='green')

from scipy.ndimage import gaussian_filter1d
xg = gaussian_filter1d(x, 5)
yg = gaussian_filter1d(y, 5)

plt.scatter(xg, yg, s=5, color='black')
plt.show()
# Save the DataFrame to a CSV file
df.to_csv('intersection_points_new.csv', index=False)
