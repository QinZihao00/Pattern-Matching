import argparse
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist


class TemplateMatchingClass(object):
    def __init__(self, target, img):
        self.target = target
        self.imgOri = img.copy()

    def getContours(self, img):
        """Get contour of template or sample image"""
        # according to opencv documentation, binary input is preferred, which can be get by applying threshold or
        # canny edge detection

        # thresholding method
        # Convert original image to gray scale
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Convert gray scale to binary, words are white, background is black, easy for findContour function
        threshold, img_binary = cv2.threshold(imgray, 120, 255, cv2.THRESH_BINARY_INV)

        plt.figure()
        plt.imshow(img_binary, cmap='gray')
        plt.show()

        if self.target == '2':
            # image morphology methods
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(img_binary, kernel, iterations=1)
            # Get Contour
            contours = cv2.Canny(img_binary, threshold1=60, threshold2=150)
            # contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(img_binary, kernel, iterations=2)
            contours = cv2.Canny(img_binary, threshold1=60, threshold2=150)
            # contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # plt.figure()
        # plt.imshow(contours, cmap='gray')
        # plt.show()
        num_components, labels = cv2.connectedComponents(contours, connectivity=8)

        # organize those points in a clockwise direction
        def clockwiseangle_and_distance(point, origin, refvec):
            # Vector between point and the origin: v = p - o
            vector = [point[0] - origin[0], point[1] - origin[1]]
            # Length of vector: ||v||
            lenvector = math.hypot(vector[0], vector[1])
            # If length is zero there is no angle
            if lenvector == 0:
                return -math.pi, 0
            # Normalize vector: v/||v||
            normalized = [vector[0] / lenvector, vector[1] / lenvector]
            dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
            diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
            angle = math.atan2(diffprod, dotprod)
            # Negative angles represent counter-clockwise angles so we need to subtract them
            # from 2*pi (360 degrees)
            if angle < 0:
                return 2 * math.pi + angle, lenvector
            # I return first the angle because that's the primary sorting criterium
            # but if two vectors have the same angle then the shorter distance should come first.
            return angle, lenvector

        def order_points(pts):
            # sort the points based on their x-coordinates
            xSorted = pts[np.argsort(pts[:, 0]), :]
            # grab the left-most and right-most points from the sorted
            # x-roodinate points
            leftMost = xSorted[:2, :]
            rightMost = xSorted[2:, :]
            # now, sort the left-most coordinates according to their
            # y-coordinates so we can grab the top-left and bottom-left
            # points, respectively
            leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
            (tl, bl) = leftMost
            # now that we have the top-left coordinate, use it as an
            # anchor to calculate the Euclidean distance between the
            # top-left and right-most points; by the Pythagorean
            # theorem, the point with the largest distance will be
            # our bottom-right point
            D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
            (br, tr) = rightMost[np.argsort(D)[::-1], :]
            # return the coordinates in top-left, top-right,
            # bottom-right, and bottom-left order
            return np.array([tl, tr, br, bl], dtype="float32")

        def order_points_self(pts, origin):
            pts = np.delete(pts, np.where(pts == origin), axis=0)
            sorted_points = [origin]
            cur_point = origin
            cur_x, cur_y = cur_point
            while len(pts) != 0:
                neighbor_found = False
                dis = 1
                while not neighbor_found:
                    neighbors = {
                        'N': [cur_x, cur_y + dis],
                        'NE': [cur_x + dis, cur_y + dis],
                        'E': [cur_x + dis, cur_y],
                        'SE': [cur_x + dis, cur_y - dis],
                        'S': [cur_x, cur_y - dis],
                        'SW': [cur_x - dis, cur_y - dis],
                        'W': [cur_x - dis, cur_y],
                        'WE': [cur_x - dis, cur_y + dis]
                    }
                    diff = pts - (np.repeat(cur_point, pts.shape[0], axis=0).reshape(-1, 2))
                    cur_maht_dis = np.linalg.norm(diff, ord=1, axis=1)
                    sorted_points.append(pts[cur_maht_dis.argmin(-1)])
                    cur_point = pts[cur_maht_dis.argmin(-1)]
                    cur_x, cur_y = cur_point
                    pts = np.delete(pts, cur_maht_dis.argmin(-1), axis=0)
                    neighbor_found = True

                    # # print(neighbors, cur_point)
                    # for direction in list(neighbors.keys()):
                    #     if neighbors[direction] in pts:
                    #         sorted_points.append(neighbors[direction])
                    #         cur_point = neighbors[direction]
                    #         cur_x, cur_y = cur_point
                    #         pts.remove(neighbors[direction])
                    #         neighbor_found = True
                    #         break
                    # dis += 1
                    # if dis % 20 == 0:
                    #     print(dis)
            return sorted_points

        # plt.figure()
        # plt.imshow((labels != 0), cmap='gray')
        # plt.show()
        # out_contour = []
        # for i in range(1, num_components):
        #     tmp = np.where(labels == i)
        #     tmp_arr = np.asarray(tmp).transpose(1, 0)
        #     tmp_arr[:, [0, 1]] = tmp_arr[:, [1, 0]]  # swap columns
        #     tmp_arr[:, 1] = tmp_arr[:, 1]
        #     out_contour.append(tmp_arr)

        # plt.figure()
        # for tmp in out_contour:
        #     plt.scatter(tmp[:, 0], - tmp[:, 1], s=1, cmap='gray')
        # plt.show()

        # plt.figure()
        # print(len(out_contour))
        # for tmp in out_contour:
        #     # tmp = tmp.tolist()
        #     print(tmp.shape)
        #     ordered_points = order_points_self(tmp, tmp[0])
        #     for i in range(len(ordered_points)):
        #         plt.scatter(ordered_points[i][0], - ordered_points[i][1], s=i * 2, cmap='gray')
        # plt.show()

        plt.figure()
        plt.imshow(contours, cmap='gray')
        plt.show()

        contours, hierarchy = cv2.findContours(contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        plt.figure()
        for i in range(len(contours)):
            plt.scatter(contours[i][:, 0, 0], -contours[i][:, 0, 1], s=1)
        plt.show()
        return [tmp for tmp in contours if (300 <= tmp.shape[0])]

    def getCV(self, img, coords=None):
        """Get complex vectors"""
        # get region of interests
        if coords is None:
            # not specified, define as the overall image
            coords = (0, 0, img.shape[1], img.shape[0])

        ROI = img[coords[1]:coords[1] + coords[3], coords[0]:coords[0] + coords[2]]
        # Find contours in that region
        contours = self.getContours(ROI)
        out_contour = []
        out_CV = []
        for contour in contours:
            contourCV = []
            x, y, w, h = cv2.boundingRect(contour)  # rectangle area bounding contour
            # cv2.rectangle(imgOri, (x, y), (x + w, y + h), (100, 100, 100), 1)
            for point in contour:
                # -x and -y are to make left and upper boundary start from 0
                contourCV.append(complex(point[0][0] - x, (point[0][1] - y)))
            out_CV.append(contourCV)
            out_contour.append(contour)
        return out_contour, out_CV

    def getFD(self, CVs):
        """Apply fourier transform on complex vectors to get fourier descriptor"""
        FDs = []
        for CV in CVs:
            FDs.append(np.fft.fft(CV))
        return FDs

    def rotationInvariant(self, fourierDesc):
        """Make fourier descriptor invariant to rotation and start point"""
        for index, value in enumerate(fourierDesc):
            fourierDesc[index] = np.absolute(value)  # Consider only absolute value
        return fourierDesc

    def scaleInvariant(self, fourierDesc):
        """Make fourier descriptor invariant to scale"""
        firstVal = fourierDesc[1]

        for index, value in enumerate(fourierDesc):
            fourierDesc[index] = value / firstVal  # Divided by first value
        return fourierDesc

    def transInvariant(self, fourierDesc):
        """Make fourier descriptor invariant to translation"""
        return fourierDesc[1:]  # Drop the first coefficient

    def getLowFreqFDs(self, fourierDesc):
        """Get the lowest X of frequency values from the fourier values."""
        # frequency order returned by np.fft is (0, 0.1, 0.2, 0.3, ...... , -0.3, -0.2, -0.1)
        # In transInvariant(), already remove first FD(0 frequency)
        return fourierDesc[:300]

    def finalFD(self, fourierDesc):
        """Process fourier descriptors"""
        cur_fourierDesc = self.rotationInvariant(fourierDesc)
        cur_fourierDesc = self.scaleInvariant(cur_fourierDesc)
        cur_fourierDesc = self.transInvariant(cur_fourierDesc)
        cur_fourierDesc = self.getLowFreqFDs(cur_fourierDesc)
        return cur_fourierDesc

    # Core match function
    def match(self, template_FD, sample_FDs, sample_contour, simThreshold, disThreshold):
        """Find match patterns and plot on original image"""
        template_FD = self.finalFD(template_FD[0])
        # dist store the distance, same order as spContours
        dist = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        for spFD in sample_FDs:
            tmp = spFD.copy()
            spFD = self.finalFD(spFD)
            # Calculate Euclidean distance between template and sample
            tmp_arr = np.array(template_FD)
            sam_arr = np.array(spFD)
            cos_sim = np.dot(tmp_arr, sam_arr) / (np.linalg.norm(tmp_arr) * np.linalg.norm(sam_arr))

            linf_norm = np.linalg.norm(tmp_arr - sam_arr, ord=np.inf)
            l2_norm = np.linalg.norm(tmp_arr - sam_arr, ord=2)
            l1_norm = np.linalg.norm(tmp_arr - sam_arr, ord=1)
            l0_norm = np.linalg.norm(tmp_arr - sam_arr, ord=0)

            metrics = {
                'cosine similarity': cos_sim,
                'l inf norm': linf_norm,
                'l2 norm': l2_norm,
                'l1 norm': l1_norm,
                'l0 norm': l0_norm
            }

            dist.append(metrics['cosine similarity'])

            x, y, w, h = cv2.boundingRect(sample_contour[len(dist) - 1])
            cv2.putText(self.imgOri, str(len(tmp)), (x, y), font, 3, (0, 0, 0), 1, cv2.LINE_AA)
            cnt = 0
            for metric_name in list(metrics.keys()):
                # Draw distance on image
                distText = '{}: {}'.format(metric_name, str(round(metrics[metric_name], 2)))
                cv2.putText(self.imgOri, distText, (x, y - 10 - 10 * cnt), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cnt += 1
            # if metrics matches, it will be good match
            if metrics['cosine similarity'] >= simThreshold and metrics['l inf norm'] <= disThreshold:
                cv2.rectangle(self.imgOri, (x - 5, y - 5), (x + w + 5, y + h + 5), (40, 255, 0), 2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Which template to match?')
    parser.add_argument('--target', default='2', type=str, help='Which one to match, 2 or C?')
    args = parser.parse_args()

    target = args.target.lower()
    print("Shape Matching of {} Using Fourier Descriptor".format(target))

    imgOri = cv2.imread(r"a4.bmp", 1)
    imgOricpy = imgOri.copy()

    tempMatch = TemplateMatchingClass(target=target, img=imgOri)

    # img_2 = cv2.imread("sample_2.jpg")
    # img_c = cv2.imread("sample_c.jpg")

    if target == 'c':
        simThreshold = 0.95
        disThreshold = 0.22
        rect = (489, 189, 134, 193)  # Rectangle area for C
    elif target == '2':
        simThreshold = 0.95
        disThreshold = 0.16
        rect = (64, 195, 171, 250)
    else:
        raise NotImplementedError("Only template matching of 2 & c supported!")

    # Get complex vector
    template_contour, template_CV = tempMatch.getCV(imgOricpy, rect)
    sample_contour, sample_CV = tempMatch.getCV(imgOricpy)
    # Get fourier descriptor
    template_FD = tempMatch.getFD(template_CV)
    sample_FDs = tempMatch.getFD(sample_CV)

    # real match function
    tempMatch.match(template_FD, sample_FDs, sample_contour, simThreshold, disThreshold)

    # Visualize: Original Image
    plt.figure(figsize=(20, 20))
    plt.imshow(tempMatch.imgOri)
    plt.show()
    #
    # # Visualize: template contour
    # template_contour_np = np.concatenate(template_contour)[:, 0]
    # plt.figure()
    # plt.scatter(template_contour_np[:, 0], -template_contour_np[:, 1], s=1)
    # plt.show()
    #
    # Visualize: sample contour
    sample_contour_np = np.concatenate(sample_contour)[:, 0]
    plt.figure()
    plt.scatter(sample_contour_np[:, 0], -sample_contour_np[:, 1], s=1)
    plt.show()
