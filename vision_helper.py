import matplotlib
# matplotlib.use('TkAgg')

import cv2
import numpy as np
import matplotlib.pyplot as plt

def np_sig(x):
    return 1 / (1 + np.exp(-x))

def normal_rgb(bgr_img):
    nm_bgr = np.zeros(bgr_img.shape)

    b = bgr_img[:, :, 0]
    g = bgr_img[:, :, 1]
    r = bgr_img[:, :, 2]

    nm_bgr[:, :, 0] = np_sig(b / (b + g + r + 1))
    nm_bgr[:, :, 1] = np_sig(g / (b + g + r + 1))
    nm_bgr[:, :, 2] = np_sig(r / (b + g + r + 1))

    return nm_bgr

def background_sub(image, background, thresh):
    img_sub = np.absolute(image - background) > thresh
    if len(image.shape) > 2:
        img_sub = np.logical_or(
            np.logical_or(img_sub[:, :, 0], img_sub[:, :, 1]),
            img_sub[:, :, 2]
        )
    return img_sub.astype(np.uint8)*255

def dense_flow(prev_bgr, current_bgr):
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(prev_bgr)
    hsv[..., 1] = 255

    next_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def sparse_flow(prev_bgr, current_bgr):

    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=100,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)#100, 0.01, 10, 3)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_bgr)

    next_gray = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)

    if p0 is None:
        return next_gray

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        current_bgr = cv2.circle(current_bgr, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(current_bgr, mask)

    return img


def show_colorspaces(img_rgb):
    f, axarr = plt.subplots(6, 3, figsize=(15, 15))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2LAB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    image_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2YUV)
    image_hls = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HLS)
    image_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV)

    axarr[0, 0].imshow(img_rgb)
    axarr[0, 0].set_title('rgb')
    axarr[0, 1].imshow(img_gray, cmap='gray')
    axarr[0, 1].set_title('grayscale')

    axarr[1, 0].imshow(img_hsv[:, :, 0], cmap='gray')
    axarr[1, 0].set_title('HSV- H channel')
    axarr[1, 1].imshow(img_hsv[:, :, 1], cmap='gray')
    axarr[1, 1].set_title('HSV- S channel')
    axarr[1, 2].imshow(img_hsv[:, :, 2], cmap='gray')
    axarr[1, 2].set_title('HSV- V channel')

    axarr[2, 0].imshow(img_lab[:, :, 0], cmap='gray')
    axarr[2, 0].set_title('LAB- L channel')
    axarr[2, 1].imshow(img_lab[:, :, 1], cmap='gray')
    axarr[2, 1].set_title('LAB- A channel')
    axarr[2, 2].imshow(img_lab[:, :, 2], cmap='gray')
    axarr[2, 2].set_title('LAB- B channel')

    axarr[3, 0].imshow(image_yuv[:, :, 0], cmap='gray')
    axarr[3, 0].set_title('YUV- Y channel')
    axarr[3, 1].imshow(image_yuv[:, :, 1], cmap='gray')
    axarr[3, 1].set_title('YUV- U channel')
    axarr[3, 2].imshow(image_yuv[:, :, 2], cmap='gray')
    axarr[3, 2].set_title('YUV- V channel')

    axarr[4, 0].imshow(image_hls[:, :, 0], cmap='gray')
    axarr[4, 0].set_title('HLS- H channel')
    axarr[4, 1].imshow(image_hls[:, :, 1], cmap='gray')
    axarr[4, 1].set_title('HLS- L channel')
    axarr[4, 2].imshow(image_hls[:, :, 2], cmap='gray')
    axarr[4, 2].set_title('HLS- S channel')

    axarr[5, 0].imshow(image_luv[:, :, 0], cmap='gray')
    axarr[5, 0].set_title('LUV- L channel')
    axarr[5, 1].imshow(image_luv[:, :, 1], cmap='gray')
    axarr[5, 1].set_title('LUV- U channel')
    axarr[5, 2].imshow(image_luv[:, :, 2], cmap='gray')
    axarr[5, 2].set_title('LUV- V channel')

    return f, axarr

"""function that perfomrs image subtraction, 
if image subtractor is given the it uses the given subtractor"""
def get_bw_foreground(img_rgb, background_rgb=None, background_subtractor=None):

    if background_subtractor:
        # if isinstance(background_subtractor, cv2.BackgroundSubtractorMOG):
        #     bw = background_subtractor.apply(img_rgb)
        if isinstance(background_subtractor, cv2.BackgroundSubtractorMOG2):
            bw = background_subtractor.apply(img_rgb)
        # elif isinstance(background_subtractor, cv2.BackgroundSubtractorGMG):
        #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        #     bw = background_subtractor.apply(img_rgb)
        #     bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        elif background_rgb:
            thr = .2
            bw = background_sub(
                normal_rgb(img_rgb),
                normal_rgb(background_rgb),
                thr
            )
        else:
            raise ValueError("Please supply either a background image or a background subtractor object")

        # take only highest probability estimates
        bw = (bw == 255).astype(np.uint8)*255
        kernel = np.ones((3, 3), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
        kernel = np.ones((5, 5), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    return bw

# bounds bw objects by suquares and returns coordinates as x, y, w, h
def find_objects(bw, roi=None):

    # give back a localization roi, if roi image was given
    if roi is not None:
        roi_copy = roi.copy()

    param_arealower = 5000*((roi.shape[1]*roi.shape[2])/(640*480))
    param_areaupper = 100000*((roi.shape[1]*roi.shape[2])/(640*480))

    # test blobs, and retrieve only the ones that look like objects (size)
    im2, cnts, hierarchy = cv2.findContours(bw.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # get largest five contour area
    coords = []
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            break
        if area > param_arealower and area < param_areaupper:
            x, y, w, h = cv2.boundingRect(cnt)
            if roi is not None:
                cv2.rectangle(roi_copy, (x, y), (x + w, y + h), (0, 255, 0), 4)
            coords += [(x, y, w, h)]
        else:
            pass
    if roi is not None:
        return roi_copy, coords
    else:
        return None, coords


# bounds boxes by boundaries and returns minimal bounding rects
def bound_contours(mask, roi=None):
    """
        returns modified roi(non-destructive) and rectangles that founded by the algorithm.
        @roi region of interest to find contours
        @return (roi, rects)
    """

    # give back a localization roi, if roi image was given
    if roi is not None:
        roi_copy = roi.copy()

    param_arealower = 5000*((mask.shape[0]*mask.shape[1])/(640*480))
    param_areaupper = 80000*((mask.shape[0]*mask.shape[1])/(640*480))

    # Find contours for detected portion of the image
    im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rects = []
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        if perimeter == 0:
            break
        if area > param_arealower and area < param_areaupper:
            rects += [cv2.minAreaRect(c)]
            box = cv2.boxPoints(rects[-1])
            box = np.int0(box)
            if roi is not None:
                cv2.drawContours(roi_copy, [box], 0, (0, 255, 0), 4)
        else:
            a=1

    if roi is not None:
        return roi_copy, rects
    else:
        return None, rects


# returns padded version of cropped image
def cropped_padder(cropped_image, width=300, height=300):

    padded_img = np.zeros((height, width, cropped_image.shape[2])).astype(np.uint8)

    pad_width = max(width - cropped_image.shape[1], 0)
    pad_height = max(height - cropped_image.shape[0], 0)

    lower_width = int(np.floor(pad_width/2))
    lower_height = int(np.floor(pad_height/2))
    upper_width = min(lower_width + cropped_image.shape[1], width)
    upper_height = min(lower_height + cropped_image.shape[0], height)

    padded_img[lower_height:upper_height, lower_width:upper_width, :] = cropped_image.copy()

    return padded_img


def crop_minAreaRect(img, rect, scale=1):
    box = cv2.boxPoints(rect)
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(scale * (x2 - x1)), int(scale * (y2 - y1)))
    if any(np.array(size) == 0):
        return None
    # cv2.circle(img_box, center, 10, (0, 255, 0), -1)  # again this was mostly for debugging purposes

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW * scale), int(croppedH * scale)),
                                       (size[0] / 2, size[1] / 2))

    return croppedRotated, center


def find_center(rect, scale=1):
    box = cv2.boxPoints(rect)

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def show_object_near_space(roi, bw_img, radius=55):
    _, rects = bound_contours(bw_img)
    for rect in rects:
        res = find_center(rect)  # center of rect
        if res is not None:
            (x, y) = res
            cv2.circle(roi, (x, y), radius, (0, 0, 255), 1)


# returns true if there is an object near the current center
def is_object_near(object_pos, other_objects_pos, radius=55):
    for other_pos in other_objects_pos:
        dist = np.sqrt(np.sum((np.array(object_pos) - np.array(other_pos))**2))
        if dist < radius:
            return True
    return False


def plt_to_img(plt_figure):
    # plt_figure.canvas.draw()

    # convert canvas to image
    img = np.fromstring(plt_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(plt_figure.canvas.get_width_height()[::-1] + (3,))

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)