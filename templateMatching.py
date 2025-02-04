import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np


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
        threshold, img_binary = cv2.threshold(imgray, 80, 255, cv2.THRESH_BINARY_INV)

        if self.target == '2':
            # image morphology methods
            kernel = np.ones((5, 5), np.uint8)
            img_dilation = cv2.dilate(img_binary, kernel, iterations=2)
            # Get Contour
            contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        else:
            kernel = np.ones((2, 2), np.uint8)
            img_erosion = cv2.erode(img_binary, kernel, iterations=2)
            contours, hierarchy = cv2.findContours(img_erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return [tmp for tmp in contours if (320 <= tmp.shape[0])]

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

            # cv2.putText(imgOri, str(len(tmp)), (x, y), font, 3, (0, 0, 0), 1, cv2.LINE_AA)
            # if metrics matches, it will be good match
            if metrics['cosine similarity'] >= simThreshold and metrics['l inf norm'] <= disThreshold:
                x, y, w, h = cv2.boundingRect(sample_contour[len(dist) - 1])
                cnt = 0
                for metric_name in list(metrics.keys()):
                    # Draw distance on image
                    distText = '{}: {}'.format(metric_name, str(round(metrics[metric_name], 2)))
                    cv2.putText(self.imgOri, distText, (x, y - 10 - 10 * cnt), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cnt += 1
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
        rect = (64, 195, 171, 232)
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
    plt.figure(figsize=(21.7, 28.0))
    plt.imshow(tempMatch.imgOri)
    plt.show()
    #
    # # Visualize: template contour
    # template_contour_np = np.concatenate(template_contour)[:, 0]
    # plt.figure()
    # plt.scatter(template_contour_np[:, 0], -template_contour_np[:, 1], s=1)
    # plt.show()
    #
    # # Visualize: sample contour
    # sample_contour_np = np.concatenate(sample_contour)[:, 0]
    # plt.figure()
    # plt.scatter(sample_contour_np[:, 0], -sample_contour_np[:, 1], s=1)
    # plt.show()
