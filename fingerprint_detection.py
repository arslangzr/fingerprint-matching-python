import cv2
import numpy as np
import os

fingerprint_test = cv2.imread("TEST_1.tif")
# cv2.imshow("Original", cv2.resize(fingerprint_test, None, fx=1, fy=1))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
match_found = False

for file in [file for file in os.listdir("database")]:
    fingerprint_database_image = cv2.imread("./database/" + file)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_test, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

    matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict()).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = min(len(keypoints_1), len(keypoints_2))
    threshold = 0.3

    if len(match_points) / keypoints > threshold:
        print("% match: ", len(match_points) / keypoints * 100)
        print("Fingerprint ID: " + str(file))
        result = cv2.drawMatches(fingerprint_test, keypoints_1, fingerprint_database_image, keypoints_2, match_points, None)
        result = cv2.resize(result, None, fx=2.5, fy=2.5)
        match_found = True
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        break

if not match_found:
    print("No matching fingerprint found in the database.")