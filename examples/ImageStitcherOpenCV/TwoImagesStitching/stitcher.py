import numpy as np
import imutils
import cv2


class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=.75, reproj_tresh=4., show_matches=False):

        image_b, image_a = images
        kps_a, features_a = self.__detect_and_describe(image_a)
        kps_b, features_b = self.__detect_and_describe(image_b)

        M = self.__match_keypoints(kps_a, kps_b, features_a, features_b, ratio,
                                   reproj_tresh)
        if M is None:
            return None

        matches, H, status = M
        result = cv2.warpPerspective(image_a, H, (
            image_a.shape[1] + image_b.shape[1], image_a.shape[0]))
        result[0:image_b.shape[0], 0:image_b.shape[1]] = image_b

        if show_matches:
            vis = self.__draw_matches(image_a, image_b, kps_a, kps_b, matches,
                                      status)
            return result, vis

        return result

    def __detect_and_describe(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            kps, features = descriptor.detectAndCompute(image, None)

        else:
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            extractor = cv2.DescriptorExtractor_create("SIFT")
            kps, features = extractor.compute(gray, kps)

        kps = np.float32([kp.pt for kp in kps])
        return kps, features

    @staticmethod
    def __match_keypoints(kps_a, kps_b, features_a, features_b, ratio,
                          reproj_tresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        raw_matches = matcher.knnMatch(features_a, features_b, 2)
        matches = []

        for m in raw_matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) > 4:
            pts_a = np.float32([kps_a[i] for (_, i) in matches])
            pts_b = np.float32([kps_b[i] for (i, _) in matches])

            H, status = cv2.findHomography(pts_a, pts_b, cv2.RANSAC,
                                           reproj_tresh)
            return matches, H, status

        return None

    @staticmethod
    def __draw_matches(image_a, image_b, keypoints_a, keypoints_b, matches,
                       status):
        height_a, width_a = image_a.shape[:2]
        height_b, width_b = image_b.shape[:2]
        vis = np.zeros((max(height_a, height_b), width_a + width_b, 3),
                       dtype='uint8')
        vis[0:height_a, 0:width_a] = image_a
        vis[0:height_b, width_a:] = image_b

        for (trainIdx, queryIdx), s in zip(matches, status):
            if s == 1:
                pt_a = (
                    int(keypoints_a[queryIdx][0]),
                    int(keypoints_a[queryIdx][1]))
                pt_b = (int(keypoints_b[trainIdx][0]) + width_a,
                        int(keypoints_b[trainIdx][1]))
                cv2.line(vis, pt_a, pt_b, (0, 255, 0), 1)

        return vis
