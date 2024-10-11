import cv2
import numpy as np
import os
from multiprocessing import Pool, cpu_count

# Load the test fingerprint image
fingerprint_test = cv2.imread("TEST_1.tif")

# SIFT detector for the test image
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(fingerprint_test, None)

# Function to serialize DMatch objects (convert them to tuples)
def serialize_match(matches):
    return [(m.queryIdx, m.trainIdx, m.distance) for m in matches]

# Function to match a single fingerprint from the database
def match_fingerprint(file):
    fingerprint_database_image = cv2.imread("./database/" + file)
    if fingerprint_database_image is None:
        return None

    # Detect keypoints and compute descriptors for the database image
    keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)

    if descriptors_1 is None or descriptors_2 is None:
        return None

    # Match descriptors using FLANN-based matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), dict())
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Filter good matches
    match_points = []
    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            match_points.append(p)

    keypoints = min(len(keypoints_1), len(keypoints_2))
    if keypoints == 0:
        return None

    threshold = 0.3
    match_ratio = len(match_points) / keypoints
    if match_ratio > threshold:
        # Serialize match points for pickling
        serialized_match_points = serialize_match(match_points)
        
        # If a match is found, return relevant information
        return {
            "file": file,
            "match_ratio": match_ratio,
            "match_points": serialized_match_points,  # Serialized match points
            "keypoints_2_len": len(keypoints_2)  # Return keypoints length for simplicity
        }
    return None

# Function to deserialize match points and draw matches
def deserialize_and_draw(fingerprint_test, keypoints_1, fingerprint_database_image, keypoints_2_len, match_points):
    # Create dummy keypoints since DMatch needs keypoint objects for visualization
    keypoints_2 = [cv2.KeyPoint() for _ in range(keypoints_2_len)]

    # Deserialize match points
    deserialized_matches = [cv2.DMatch(int(m[0]), int(m[1]), float(m[2])) for m in match_points]

    # Draw matches
    result_image = cv2.drawMatches(fingerprint_test, keypoints_1, 
                                   fingerprint_database_image, keypoints_2, 
                                   deserialized_matches, None)
    return result_image

# Function to process the results
def process_results(results):
    for result in results:
        if result is not None:
            print(f"% match: {result['match_ratio'] * 100}")
            print(f"Fingerprint ID: {result['file']}")
            
            # Load the fingerprint database image again for drawing
            fingerprint_database_image = cv2.imread("./database/" + result['file'])

            # Deserialize and draw the matches
            result_image = deserialize_and_draw(fingerprint_test, keypoints_1, fingerprint_database_image, 
                                                result['keypoints_2_len'], result['match_points'])
            result_image = cv2.resize(result_image, None, fx=2.5, fy=2.5)
            
            # Display the result (can be uncommented)
            # cv2.imshow("result", result_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return True
    return False

if __name__ == '__main__':
    # Get the list of files in the database
    files = [file for file in os.listdir("database")]

    # Create a pool of worker processes
    with Pool(cpu_count()) as pool:
        # Distribute fingerprint matching across processes
        results = pool.map(match_fingerprint, files)
    
    # Process and display the results
    if not process_results(results):
        print("No matching fingerprint found in the database.")
