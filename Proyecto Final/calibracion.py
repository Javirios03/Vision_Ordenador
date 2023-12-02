import numpy as np
import cv2

# Load your images
path = "Calib_Images/"
images = []
for i in range(16):
    image = cv2.imread(path + f"image{i}" + ".jpg")
    images.append(image)
    images.append(cv2.imread("Pattern/piano.png"))

# Define the chessboard size or custom pattern size
pattern_size = (23, 3)

# Arrays to store object points and image points from all images
object_points = []
image_points = []

# Define the 3D coordinates of the pattern
objp = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

for image in images:
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find corners or keypoints
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # If corners are found, store object points and image points
        object_points.append(objp)
        image_points.append(corners)

# Perform camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, gray.shape[::-1], None, None)

# Save the calibration parameters (mtx, dist) for later use
np.savez("calibration_params.npz", mtx=mtx, dist=dist)

# Undistort an example image
example_image = images[0]
undistorted_image = cv2.undistort(example_image, mtx, dist, None, mtx)

# Display the undistorted image
cv2.imshow("Undistorted Image", undistorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
