from stitcher import Stitcher
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=False,
                help="path to the first image")
ap.add_argument("-s", "--second", required=False, help="path to the second "
                                                       "image")
args = vars(ap.parse_args())

imageA = cv2.imread(args["first"] if args["first"] else
                    "../../../data/input/church_1.jpg")
imageB = cv2.imread(args["second"] if args["first"] else
                    "../../../data/input/church_2.jpg")

stitcher = Stitcher()
result, matches = stitcher.stitch([imageA, imageB], show_matches=True)

cv2.imwrite("../../../data/output/keypoints_matches.png", matches)
cv2.imwrite("../../../data/output/stitch_result.png", result)
