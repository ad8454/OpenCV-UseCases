import cv2
import numpy as np

x0, y0, x1, y1 = 0, 0, 0, 0
gotRect = None

def bonus():
    """
    Use Gaussian and Laplacian pyramids to compute blended image
    """
    apple = cv2.imread('apple.jpg')
    orange = cv2.imread('orange.jpg')
    level = 5
    prevGaussA = apple.copy()
    prevGaussO = orange.copy()
    pyrUpA = []
    pyrUpO = []

    # use gaussian pyramids to compute and store laplacian pyramids of the image
    for i in range(level):
        curGaussA = cv2.pyrDown(prevGaussA)
        h, w, d = prevGaussA.shape
        pyrUpA.append(cv2.subtract(prevGaussA, cv2.pyrUp(curGaussA, dstsize=(w,h))))
        prevGaussA = curGaussA

        curGaussO = cv2.pyrDown(prevGaussO)
        h, w, d = prevGaussO.shape
        pyrUpO.append(cv2.subtract(prevGaussO, cv2.pyrUp(curGaussO, dstsize=(w,h))))
        prevGaussO = curGaussO

    pyrUpA.append(prevGaussA)
    pyrUpO.append(prevGaussO)
    pyrUpA.reverse()
    pyrUpO.reverse()

    # build up final image by merging corresponding pyramids of both images
    merged = []
    for pA, pO in zip(pyrUpA, pyrUpO):
        merged.append(np.hstack((pA[:, 0:pA.shape[1]/2], pO[:, pO.shape[1]/2:])))
    for i in range(len(merged) - 1):
        h, w, d = merged[i+1].shape
        merged[i+1] = cv2.add(merged[i+1], cv2.pyrUp(merged[i], dstsize=(w, h)))

    cv2.imshow('Result 7: Bonus', merged[-1])
    cv2.waitKey(0)


def six():
    """
    Use k-means clustering to segment berries based on color
    """
    orig = cv2.imread('berries.png')
    # blur to remove details and smoothen image
    img = cv2.GaussianBlur(orig, (7, 7), 4, 4)
    # convert to HSV and saturate the colors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = np.uint16(img)
    img[:, :, 1] += 128
    img[:, :, 2] += 64
    img[img > 255] = 255
    img = np.uint8(img)
    # switch back to BGR
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    # k-means segmentation
    k = 5
    flat = np.float32(img.reshape(img.shape[0] * img.shape[1], 3))
    termCrit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
    ret, label, center = cv2.kmeans(flat, k, None, termCrit, 10, cv2.KMEANS_RANDOM_CENTERS)

    results = []
    # for each cluster
    for i in range(k):
        # extract required color cluster in binary
        mask = [[0, 0, 0]]*(k-1)
        mask.insert(i, [255, 255, 255])
        mask = np.asarray(mask)
        binary = mask[label.flatten()]
        binary = np.uint8(binary.reshape(orig.shape))
        # erode and dilate image to remove cluster fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        binary = cv2.erode(binary, kernel)
        binary = cv2.erode(binary, kernel)
        binary = cv2.dilate(binary, kernel)
        binary = cv2.dilate(binary, kernel)
        binary[binary > 0] = 255
        # keep only required pixel values
        berry = orig.copy()
        berry = berry * np.int32(binary)
        berry[np.where((berry == [0, 0, 0]).all(axis=2))] = [2147483647, 2147483647, 2147483647]
        if i % 2 == 0:
            results.append(berry)

    cv2.imshow('Result 6: Original Image', orig)
    cv2.waitKey(0)
    for berry in results:
        cv2.imshow('Result 6: Segmented Berries', berry)
        cv2.waitKey(0)


def getClicks(event, x, y, flags, param):
    """
    Helper method for five(). Used to capture user mouse events
    """
    global x0, y0, x1, y1, gotRect
    if event == cv2.EVENT_LBUTTONDOWN:
        x0, y0 = x, y
        x1, y1 = x, y
        gotRect = False
    elif event == cv2.EVENT_MOUSEMOVE:
        x1, y1 = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        gotRect = True


def five():
    """
    Use meanshift algorithm to track ball. Implementation referenced
    from http://docs.opencv.org/3.1.0/db/df8/tutorial_py_meanshift.html
    """
    global x0, y0, x1, y1, isDragging
    cap = cv2.VideoCapture('trackball.avi')
    res, orig = cap.read()
    frame1 = orig
    title = 'Result 5: Drag mouse to draw a box around the red ball'
    cv2.namedWindow(title)
    # set event listener and capture user coordinates
    cv2.setMouseCallback(title, getClicks)
    print('Drag mouse to draw a box around the red ball')
    while gotRect != True:
        cv2.imshow(title, frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if gotRect == False:
            frame1 = np.copy(orig)
            cv2.rectangle(frame1, (x0, y0), (x1, y1), (250, 0, 0), 2)
    cv2.destroyAllWindows()

    # isolate user defined rectangle and calculate its histogram in HSV space
    x = min(x0, x1)
    y = min(y0, y1)
    roi = (x, y, abs(x0-x1), abs(y0-y1))
    ball = frame1[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
    ball = cv2.cvtColor(ball, cv2.COLOR_BGR2HSV)
    range = cv2.inRange(ball, np.asarray([0., 60., 30.], np.uint8), np.asarray([180., 255., 255.], np.uint8))
    ballHist = cv2.calcHist([ball], [0], range, [180], [0, 180])
    cv2.normalize(ballHist, ballHist, 0, 255, cv2.NORM_MINMAX)

    # define termination criteria
    termCrit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    res, frame1 = cap.read()
    while res:
        # use back projection to compute meanshift
        ballHsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        res = cv2.calcBackProject([ballHsv], [0], ballHist, [0, 180], 1)
        res, roi = cv2.meanShift(res, roi, termCrit)
        x, y, w, h = roi
        cv2.rectangle(frame1, (x, y), (x + w, y + h), 255,2)

        cv2.imshow('Result 5', frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # get next frame
        res, frame1 = cap.read()

    cap.release()
    cv2.destroyAllWindows()


def four():
    print ('Implemented in Matlab. Please see four.m')


def three():
    """
    use Hough transform to detect circles
    """

    orig = cv2.imread('cropcirlces.png')
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # blur image to improve accuracy
    img = cv2.GaussianBlur(img, (5, 5), 2, 2)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 6, 60, None, 145., 10., 10, 40)[0]

    # display correctly identified circles in blue
    correct = 0
    for i in range(len(circles)):
        if i == 36:
            cv2.circle(orig, (circles[i][0], circles[i][1]), circles[i][2], (0, 0, 255), 2)
        elif i == 7:
            cv2.circle(orig, (circles[i][0], circles[i][1]), circles[i][2], (0, 160, 255), 2)
        else:
            cv2.circle(orig, (circles[i][0], circles[i][1]), circles[i][2], (155, 0, 0), 2)
            correct += 1

    print 'Correct: '+str(correct)
    print 'Total: '+str(len(circles))
    print 'Accuracy: '+str((correct / float(len(circles)))*100)+'%'

    cv2.imshow('Result 3', orig)
    cv2.waitKey(0)


def two():
    """
    Implementation of Wiener deconvolution as given in opencv/samples/python/deconvolution.py
    """
    img = cv2.imread('carlicense_noisy.png', 0)
    img = np.float32(img)/255.0

    d = 31
    h, w = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    img = img*w + img_blur*(1-w)

    IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
    # custom values of noise and psf found by repeatedly plugging in different values till a legible image was obtained
    noise = 0.002
    psf = np.ones((12, 12))/144
    psf /= psf.sum()
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
    PSF2 = (PSF**2).sum(-1)
    iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
    RES = cv2.mulSpectrums(IMG, iPSF, 0)
    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res = np.roll(res, -kh//2, 0)
    res = np.roll(res, -kw//2, 1)

    print 'License plate reads  HSD 4671'
    cv2.imshow('Result 2', res)
    cv2.waitKey(0)


def one():
    """
    Normalize all color values between 0 and 255
    """
    img = cv2.imread('highway.png')
    cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
    print 'Highway signs read  Walnut Creek  San Jose  Pittsburg Antioch  Martinez Hercules'
    cv2.imshow('Result 1', img)
    cv2.waitKey(0)


def main():
    while True:
        print ''
        num = input('Enter problem number (1 through 7) to see result, 0 to quit: ')
        if num == 0:
            exit(0)
        elif num == 1:
            one()
        elif num == 2:
            two()
        elif num == 3:
            three()
        elif num == 4:
            four()
        elif num == 5:
            five()
        elif num == 6:
            six()
        else:
            bonus()


if __name__ == '__main__':
    main()