import numpy as np
import imutils
import cv2


class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=.75, reproj_tresh=4., show_matches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        image_b, image_a = images
        kps_a, features_a = self.__detect_and_describe(image_a)
        kps_b, features_b = self.__detect_and_describe(image_b)

        # match features between the two images
        M = self.__match_keypoints(kps_a, kps_b, features_a, features_b, ratio,
                                   reproj_tresh)

        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama
        if M is None:
            return None

        # otherwise, apply a perspective warp to stitch the images
        # together
        matches, H, status = M
        result = cv2.warpPerspective(image_a, H, (
            image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
        result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

        # check to see if the keypoint matches should be visualized
        if show_matches:
            vis = self.__draw_matches(image_a, image_b, kps_a, kps_b, matches,
                                      status)

            # return a tuple of the stitched image and the
            # visualization
            return result, vis

        # return the stitched image
        return result

    def __detect_and_describe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            kps, features = descriptor.detectAndCompute(image, None)

        # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            kps, features = extractor.compute(gray, kps)

        # convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # return a tuple of keypoints and features
        return kps, features

    def __match_keypoints(self, kpsA, kpsB, featuresA, featuresB, ratio,
                          reprojTresh):
        # compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        # loop over raw matches
        for m in raw_matches:
            # ensure the distance is within a certain ratio of each other
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojTresh)

            # return the matches along with the homography matrix and status
            # of each matched point
            return matches, H, status

        # otherwise, no homography could be computed
        return None

    @staticmethod
    def __draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches,
                       status):
        # initialize the output visualization image
        (hA, wA) = image_a.shape[:2]
        (hB, wB) = image_b.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
        vis[0:hA, 0:wA] = image_a
        vis[0:hB, wA:] = image_b

        # loop over mathces
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # only process the match if the keypoint was successfully matched
            if s == 1:
                # draw the match
                ptA = (
                int(keypoints_a[queryIdx][0]), int(keypoints_a[queryIdx][1]))
                ptB = (int(keypoints_b[trainIdx][0]) + wA,
                       int(keypoints_b[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # return the visualization
        return vis
