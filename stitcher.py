from panorama import Stitcher
import argparse
import imutils
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True, help="path to the first image")
ap.add_argument("-s", "--second", required=True, help="path to the second "
                                                      "image")
args = vars(ap.parse_args())

imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width=4160)
imageB = imutils.resize(imageB, width=4160)

stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)

# cv2.imwrite("imageA.png", imageA)
# cv2.imwrite("imageB.png", imageB)
# cv2.imwrite("kps_matches.png", vis)
cv2.imwrite("result.png", result)
print("written", os.getcwd())
# cv2.imshow("Image A", imageA)
# cv2.imshow("Image B", imageB)
# cv2.imshow("Keypoint matches", vis)
# cv2.imshow("Result", result)
# cv2.waitKey(0)
