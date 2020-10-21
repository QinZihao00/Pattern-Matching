import numpy as np
import matplotlib.pyplot as plt
import cv2


def getContours(img):
    """Get contour of template or sample image"""
    # Convert original image to gray scale
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert gray scale to binary
    threshold, img_binary = cv2.threshold(imgray, 80, 255, cv2.THRESH_BINARY)
    # Words are white, background is black, easy for findContour function
    img_binary_inv = cv2.bitwise_not(img_binary)
    # Dilation make all 6 to form a closed loop
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(img_binary_inv, kernel, iterations=2)
    # Get Contour
    contours, hierarchy = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours


def getTemplateCV():
    """Get complex vector of template contour"""
    # Template region
    template_region = imgOricpy[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    # Automatically find template contour
    tp_contour = getContours(template_region)

    for contour in tp_contour:
        x, y, w, h = cv2.boundingRect(contour)  # rectangle area bounding contour
        for point in contour:
            # -x and -y are to make left and upper boundry start from 0
            templateComVector.append(complex(point[0][0] - x, (point[0][1] - y)))
    return tp_contour


def getSampleCV():
    """Get complex vectors of all contours on sample image"""
    sp_contours = getContours(imgOricpy)

    for contour in sp_contours:
        sample_cv = []
        x, y, w, h = cv2.boundingRect(contour)  # Rectangle area bounding contour
        cv2.rectangle(imgOri, (x, y), (x + w, y + h), (100, 100, 100), 1)

        for point in contour:
            sample_cv.append(complex(point[0][0] - x, (point[0][1] - y)))
        # sampleComVectors store complex vectors of all sample contours
        sampleComVectors.append(sample_cv)
        # sampleContours store all sample contours, same order with sampleComVectors
        sampleContours.append(contour)
    return sp_contours


def getTemplateFD():
    """Apply fourier transform on template complex vectors to get fourier descriptor"""
    return np.fft.fft(templateComVector)


def getSampleFDs():
    """Apply fourier transform on all sample complex vectors to get fourier descriptor"""
    FDs = []
    for sample_cv in sampleComVectors:
        sampleFD = np.fft.fft(sample_cv)
        FDs.append(sampleFD)

    return FDs


def rotationInvariant(fourierDesc):
    """Make fourier descriptor invariant to rotation and start point"""
    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = np.absolute(value) # Consider only absolute value

    return fourierDesc


def scaleInvariant(fourierDesc):
    """Make fourier descriptor invariant to scale"""
    firstVal = fourierDesc[0]

    for index, value in enumerate(fourierDesc):
        fourierDesc[index] = value / firstVal   # Divided by first value

    return fourierDesc


def transInvariant(fourierDesc):
    """Make fourier descriptor invariant to translation"""
    return fourierDesc[1:]  # Drop the first coefficient


def getLowFreqFDs(fourierDesc):
    """Get the lowest X of frequency values from the fourier values."""
    # frequency order returned by np.fft is (0, 0.1, 0.2, 0.3, ...... , -0.3, -0.2, -0.1)
    # In transInvariant(), already remove first FD(0 frequency)

    return fourierDesc[:4]


def finalFD(fourierDesc):
    """Process fourier descriptors"""
    fourierDesc = rotationInvariant(fourierDesc)
    fourierDesc = scaleInvariant(fourierDesc)
    fourierDesc = transInvariant(fourierDesc)
    fourierDesc = getLowFreqFDs(fourierDesc)

    return fourierDesc


# Core match function
def match(template_FD, sample_FDs):
    """Find match patterns and plot on original image"""
    template_FD = finalFD(template_FD)
    # dist store the distance, same order as spContours
    dist = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for spFD in sample_FDs:
        spFD = finalFD(spFD)
        # Calculate Euclidean distance between template and sample
        dist.append(np.linalg.norm(np.array(spFD) - np.array(template_FD)))
        x, y, w, h = cv2.boundingRect(sampleContours[len(dist) - 1])
        # Draw distance on image
        distText = str(round(dist[len(dist) - 1], 2))
        cv2.putText(imgOri, distText, (x, y - 8), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # if distance is less than threshold, it will be good match
        if dist[len(dist) - 1] < distThreshold:
            cv2.rectangle(imgOri, (x - 5, y - 5), (x + w + 5, y + h + 5), (40, 255, 0), 2)


if __name__ == '__main__':
    print("Shape Matching Using Fourier Descriptor")

    imgOri = cv2.imread(r"a4.bmp", 1)
    imgOricpy = imgOri.copy()
    img_2 = cv2.imread("sample_2.jpg")
    img_c = cv2.imread("sample_c.jpg")

    distThreshold = 0.06
    rect = (64, 195, 171, 232)      # Rectangle area for 2
    # rect = (489, 189, 134, 193)   # Rectangle area for C
    templateComVector = []
    sampleComVectors = []
    sampleContours = []

    # Get complex vector
    template_contour = getTemplateCV()
    sample_contour = getSampleCV()
    # Get fourier descriptor
    template_FD = getTemplateFD()
    sampleFDs = getSampleFDs()
    # real match function
    match(template_FD, sampleFDs)

    # Visualize: Original Image
    plt.figure()
    plt.imshow(imgOri)
    plt.show()

    # Visualize: template contour
    template_contour_np = np.concatenate(template_contour)[:, 0]
    plt.figure()
    plt.scatter(template_contour_np[:, 0], -template_contour_np[:, 1])
    plt.show()

    # Visualize: sample contour
    sample_contour_np = np.concatenate(sample_contour)[:, 0]
    plt.figure()
    plt.scatter(sample_contour_np[:, 0], -sample_contour_np[:, 1])
    plt.show()

