import cv2
import numpy as np

# Read three images - two input images and a ground truth image

our = cv2.imread('final/new_robot.png', cv2.IMREAD_COLOR)
theirs = cv2.imread('final/other_robot.png', cv2.IMREAD_COLOR)

gt = cv2.imread('ground_truth/robot.png', cv2.IMREAD_COLOR)

our = cv2.resize(our, (128, 128))
theirs = cv2.resize(theirs, (128, 128))
gt = cv2.resize(gt, (128, 128))

# Save the images
our2 = cv2.resize(our, (1080, 1080), interpolation=cv2.INTER_LINEAR)
cv2.imwrite('final/new_robot_1080.png', our2)

# Convert to float32
our = np.float32(our) / 255.0
theirs = np.float32(theirs) / 255.0
gt = np.float32(gt) / 255.0

# Calculate the L2 relative error between the two images
our_l2 = np.mean((our - gt) ** 2)
their_l2 = np.mean((theirs - gt) ** 2)

print('Our L2: {}'.format(our_l2))
print('Their L2: {}'.format(their_l2))