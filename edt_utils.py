
from matplotlib import pyplot as plt
from itertools import groupby
from scipy import interpolate
from scipy import ndimage
from PIL import Image
import skimage as ski
import pandas as pd
import numpy as np
import cv2 as cv
import scipy
import sys
#import pytesseract
import pprint
import math
import ss

def laplacian_filter(img, kSize=3, gSize=3, alpha=1.0):
    input_is_bgr = len(img.shape) == 3

    if input_is_bgr: 
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = img

    # Gaussian blurring / low-pass filter.
    gauss = cv.GaussianBlur(gray_img, (gSize, gSize), 0.0)

    # Edge detection / high-pass filter.
    lpl = cv.Laplacian(gauss, cv.CV_32F, ksize=kSize)
    if input_is_bgr: 
        lpl  = cv.cvtColor(lpl, cv.COLOR_GRAY2BGR)

    # Image sharpening.
    filtered_img = img.astype("float32") - alpha * lpl
    
    return np.clip(filtered_img, 0.0, 255.0).astype("uint8")

def extract_sequence(cropped_signal, kSize2d=3, kSize1d=3):
    """
    Based off https://github.com/alphanumericslab/ecg-image-kit.
    """

    # BGR to grayscale conversion.
    if len(cropped_signal == 3):
        gray_signal = cv.cvtColor(cropped_signal, cv.COLOR_BGR2GRAY)
    else:
        gray_signal = cropped_signal

    # Image sharpening.
    lpl_signal = laplacian_filter(gray_signal, 3, 7, 0.5)

    # Simple argmin.
    argmin_seq = lpl_signal.argmin(axis=0)

    # 1D blur.
    box1d_seq = cv.filter2D(lpl_signal, cv.CV_32F, np.ones((1, kSize1d), np.float32) / kSize1d).argmin(axis=0)

    # Box blur.
    box2d_seq = cv.filter2D(lpl_signal, cv.CV_32F, np.ones((kSize2d, kSize2d), np.float32) / (kSize2d * kSize2d)).argmin(axis=0)

    # All left/right neighbors.
    h0 = np.array([[1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0],
                   [1.0, 0.0, 1.0]], np.float32) / 7.0
    lr_neigh_seq = cv.filter2D(lpl_signal, cv.CV_32F, h0).argmin(axis=0)

    # All combined neighbours.    
    h1 = np.array([1.0, 1.0, 1.0], np.float32) / 3.0
    z1 = cv.filter2D(lpl_signal, cv.CV_32F, h1).argmin(axis=0)

    h2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], np.float32) / 3.0
    z2 = cv.filter2D(lpl_signal, cv.CV_32F, h2).argmin(axis=0)

    h3 = np.array([[0.0, 0.0, 1.0], 
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]], np.float32) / 3.0
    z3 = cv.filter2D(lpl_signal, cv.CV_32F, h3).argmin(axis=0)

    all_neigh_seq = np.maximum(np.maximum(z1, z2), z3)

    output_seq = np.median([argmin_seq,
                            box1d_seq,
                            box2d_seq,
                            lr_neigh_seq,
                            all_neigh_seq
                            ], axis=0)
    return output_seq

def extract_image(cropped_img, kSize2d=3, kSize1d=3):
    """
    Based off https://github.com/alphanumericslab/ecg-image-kit.
    """

    # BGR to grayscale conversion.
    if len(cropped_img == 3):
        gray_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2GRAY)
    else:
        gray_img = cropped_img

    # Image sharpening.
    lpl_img = laplacian_filter(gray_img, 3, 7, 0.5)

    
    # 1D blur.
    box1d_img = cv.filter2D(lpl_img, cv.CV_32F, np.ones((1, kSize1d), np.float32) / kSize1d)

    # Box blur.
    box2d_img = cv.filter2D(lpl_img, cv.CV_32F, np.ones((kSize2d, kSize2d), np.float32) / (kSize2d * kSize2d))

    # All left/right neighbors.
    h0 = np.array([[1.0, 0.0, 1.0],
                   [1.0, 1.0, 1.0],
                   [1.0, 0.0, 1.0]], np.float32) / 7.0
    lr_neigh_img = cv.filter2D(lpl_img, cv.CV_32F, h0)

    # All combined neighbours.    
    h1 = np.array([1.0, 1.0, 1.0], np.float32) / 3.0
    z1 = cv.filter2D(lpl_img, cv.CV_32F, h1)

    h2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0]], np.float32) / 3.0
    z2 = cv.filter2D(lpl_img, cv.CV_32F, h2)

    h3 = np.array([[0.0, 0.0, 1.0], 
                   [0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0]], np.float32) / 3.0
    z3 = cv.filter2D(lpl_img, cv.CV_32F, h3)

    all_neigh_img = np.maximum(np.maximum(z1, z2), z3)

    output_img = np.median([lpl_img,
                            box1d_img,
                            box2d_img,
                            lr_neigh_img,
                            all_neigh_img
                            ], axis=0)
    
    output_img = cv.normalize(output_img, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    return output_img


def is_nan(value):
    try:
        return math.isnan(float(value))
    except ValueError:
        return False

def get_rectangular_contours(contours):
    """Approximates provided contours and returns only those which have 4 vertices"""
    res = []
    for contour in contours:
        hull = cv.convexHull(contour)
        peri = cv.arcLength(hull, closed=True)
        approx = cv.approxPolyDP(hull, 0.04 * peri, closed=True)
        if len(approx) == 4:
            res.append(approx)
    return res

def py_blockproc(A, blockdims, func=0):
    '''
     py_blochproc : process block using a function
     A: original image or matrix
     blockdims: dimension of the blocks 
     func= name of the function 
    '''
    #TODO: pass the function instead of hard coding
    vr, hr = A.shape[0] // blockdims[0], A.shape[1] // blockdims[1]
    B = np.zeros((vr,hr))

    verts = np.vsplit(A, vr)
    for i in range(len(verts)):
       for j, v in enumerate(np.hsplit(verts[i], hr)):
          B[i,j]=(np.std(A[
             i * blockdims[0] : (i + 1) * blockdims[0],
             j * blockdims[1] : (j + 1) * blockdims[1]
            ])) # Calculate the standard deviation of the block 
    return B #retruns 

def display_segments(name, item, axis='off', plot_callback=None):
    plt.figure(figsize=(12, 9))
    plt.imshow(item, cmap="magma")
    plt.title(name)
    plt.axis(axis)
    plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)
    if plot_callback:
        plot_callback(plt.gcf())
        plt.close('all')
    else:
        plt.show()
        
import numpy as np
from scipy import ndimage

def get_values_from_img(
    roi,
    extrapolate_edges=True
):
    """
    Track a signal in a binary image column-by-column, handle gaps,
    interpolate missing points, and optionally extrapolate edges.

    Parameters
    ----------
    roi : 2D numpy array
        Binary image (signal = nonzero)
    extrapolate_edges : bool
        If True, extrapolate y-values before the first and after the last detection.

    Returns
    -------
    xs_full : np.ndarray
        All column indices (0 .. width-1)
    ys_full : np.ndarray
        Continuous y-values (interpolated + extrapolated)
    """

    height, width = roi.shape[:2]
    bool_roi = roi != 0 # boolean version of the signal (True = signal)

    xs = []
    ys = []

    old_dy_dx = 0.0

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = np.abs(array - value).argmin()
        return array[idx]

    # --- 1. Track the signal column-by-column ---
    for x, col in enumerate(bool_roi.T):

        # Find white pixels in this column
        pixel_rows = np.where(col)[0] # remember np.where returns a TUPPLE of arrays , so we need the [0] to get the actual array

        if pixel_rows.size == 0:
            # No signal in this column → skip (gap)
            continue

        # Convert to bottom-origin coordinate system
        pixel_loc = height - pixel_rows

        if len(xs) == 0:
            # First detection → just take the median
            ys.append(np.median(pixel_loc))
            xs.append(x)
            continue

        # Label connected components in this column
        labels, num = ndimage.label(col, structure=np.ones((3,)))

        if num == 0:
            continue

        median_list = []
        dy_dx_list = []
        d2y_dx2_list = []

        for n in range(1, num + 1):
            blob_rows = np.where(labels == n)[0]
            blob_loc = height - blob_rows

            median_pixel = find_nearest(blob_loc, np.median(blob_loc))
            median_list.append(median_pixel)

            dy_dx = median_pixel - ys[-1]
            d2y_dx2 = dy_dx - old_dy_dx

            dy_dx_list.append(dy_dx)
            d2y_dx2_list.append(d2y_dx2)

        # Choose blob with smallest curvature change
        best = np.argmin(np.abs(d2y_dx2_list))

        old_dy_dx = dy_dx_list[best]
        xs.append(x)
        ys.append(median_list[best])

    xs = np.array(xs)
    ys = np.array(ys)

    # --- 2. Interpolate missing columns ---
    xs_full = np.arange(width)

    if xs.size == 0:
        # No signal at all
        return xs_full, np.zeros_like(xs_full)

    ys_full = np.interp(xs_full, xs, ys)

    # --- 3. Optional: extrapolate edges ---
    if extrapolate_edges:
        # Before first detection
        ys_full[:xs[0]] = ys[0] - old_dy_dx * (xs[0] - xs_full[:xs[0]])

        # After last detection
        ys_full[xs[-1]:] = ys[-1] + old_dy_dx * (xs_full[xs[-1]:] - xs[-1])

    return height, width, xs_full, ys_full

# def get_values_from_img(roi):
#     '''
#     get the values of coord x and y for the image that contain the signal
#     INPUT:
#         roi: binary image with signal in white
#     OUTPUT:
#         xs, ys: values of the signal
#     '''
#     def find_nearest(array, value):
#         array = np.asarray(array)
#         idx = (np.abs(array - value)).argmin()
#         return array[idx]
#     width, length = roi.shape[:2]
#     xs, ys = [], []
#     bool_roi = roi != 0 # boolean version of the signal (True = signal)
#     old_dy_dx = 0.0 # store the previous slope of the signal
#     for i, col in enumerate(bool_roi.T): # Transpose the image so the interaction goes column by column
#         if len(xs) != 0:
#             label, num = ndimage.label(col, structure=np.ones((3,))) #label the connected components in the column
#             if num != 0:
#                 median_list = []
#                 dy_dx_list = []
#                 d2y_dx2_list = []
#                 for n in range(1, num + 1):
#                     pixel_loc = width - np.where(label == n)[0] # remember np.where returns a TUPPLE of arrays , so we need the [0] to get the actual array
#                     median_pixel = find_nearest(pixel_loc, np.median(pixel_loc))
#                     median_list.append(median_pixel)
#                     dy_dx = (median_pixel - ys[-1]) #/ (i - xs[-1]), ys[-1] is the last element
#                     d2y_dx2 = (dy_dx - old_dy_dx) #/ (i - xs[-1])
#                     dy_dx_list.append(dy_dx)
#                     d2y_dx2_list.append(d2y_dx2)
#                 tmp = np.argmin(np.abs(d2y_dx2_list))
#                 old_dy_dx = dy_dx_list[tmp]
#                 xs.append(i)
#                 ys.append(median_list[tmp])
#         else: # fist column with a signal
#             pixel_loc = width - np.where(col)[0]
#             if pixel_loc.size > 0:
#                 median_pixel = np.median(pixel_loc)
#                 xs.append(i)
#                 ys.append(median_pixel)
#     return width, length, xs, ys

import numpy as np

def measure_extract_pulse(x, y, verbose=0):
    # Compute pulse height
    min_pulse = np.min(y)
    max_pulse = np.max(y)
    height = max_pulse - min_pulse

    # Threshold at half-height
    threshold = min_pulse + height / 2

    # Indices where signal is above half-height
    index = np.where(y >= threshold)[0]

    # Pulse width in x-units
    width = x[index[-1]] - x[index[0]]

    if verbose:
        print(f"pulse height: {height}")
        print(f"pulse width: {width} time units")

    return width, height


# def measure_extract_pulse(x, y, verbose=0):
#     min_pulse = np.min(y)
#     max_pulse = np.max(y)

#     height = np.max(max_pulse-min_pulse)
#     threshold = height / 2
#     index = np.where((y - min_pulse)>=threshold)[0]
#     width = x[index[-1]] - x[index[0]]
#     if verbose > 0:
#         print(f"pulse height: {height}")
#         print(f"pulse width: {width} time units")
#     return width, height

def convert_to_secmv(xs, ys, wp, hp, ws, baseline, pulse_per_sec, pulse_per_mv):
    '''
    INPUTS:
        xs: x-axis in pts
        ys: y-axis in pts
        wp: pulse width in pts
        hp: pulse height in pts
        baseline: segment baseline in pts
        ws:  segment width in pts
    '''
    zero_line = ws - baseline
    ymv = (ys - zero_line) / (hp * pulse_per_mv)
    sec_per_pts = (pulse_per_sec / wp)
    xsec = sec_per_pts * np.asarray(xs)
    return xsec, ymv


def detect_rotated_template(image, template, fill_color):
    # Initialize ORB detector
    orb = cv.ORB_create(nfeatures=2000)
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(image, None)

    # Use Brute-Force matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    # We need at least 4 matches to define a perspective transform
    if len(matches) > 10:
        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the transformation matrix (Homography)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        
        if M is not None:
            # Get corners of the template
            h, w = template.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # Transform corners to the image space
            dst = cv.perspectiveTransform(pts, M)

            # Draw and fill the detected polygon
            processed_img = image.copy()
            cv.fillPoly(processed_img, [np.int32(dst)], fill_color)
            
            return processed_img, True

    return image, False

def detect_ref_pulse(roi, template, location='right', threshold=0.6, verbose=2):
    # 1. Dimensions and Early Exit
    h_roi, w_roi = roi.shape[:2]
    h_tmpl, w_tmpl = template.shape[:2]

    if h_roi <= h_tmpl or w_roi <= w_tmpl:
        return False, location, 0, 0, 0, np.nan, np.nan

    # 2. Template Matching
    res = cv.matchTemplate(roi, template, cv.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # OpenCV max_loc is (x, y) -> (Column, Row)
    # 
    y_col = max_loc[0]
    x_row = max_loc[1]
    similarity_value = max_val

    # 3. Threshold Check 
    loc = np.where(res >= threshold)
    detected = len(loc[0]) > 0

    # 4. Pulse Extraction & Measurement
    if detected:
        # Extract using the best match found by minMaxLoc
        extracted_pulse = roi[x_row : x_row + h_tmpl, y_col : y_col + w_tmpl]
        
        # Get value and measure the pulse
        _, _, xpulse, ypulse = get_values_from_img(extracted_pulse)
        wpulse, hpulse = measure_extract_pulse(xpulse, ypulse)
        
        if verbose > 1:
            print(f"INFO: {location} Pulse Detected. Sim: {similarity_value:.4f}")
    else:
        wpulse, hpulse = np.nan, np.nan

    return detected, location, similarity_value, x_row, y_col, wpulse, hpulse

# def detect_ref_pulse(roi, template,location='right', threshold=0.6, verbose=2):

#     '''

#     '''

#     if roi.shape[0] <= template.shape[0] or roi.shape[1] <= template.shape[1]:

#         #template is bigger than roi. Can not perform matchTemplate

#         empty_list=[]
#         empty_array = np.array(empty_list)
#         loc = (empty_array,empty_array )

#     else:

#         method = cv.TM_CCORR_NORMED
#         res = cv.matchTemplate(roi,template,method) # try tofind the pulse using a template match
#         min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

#         # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum

#         if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:

#             top_left = min_loc
#             x = top_left[1]
#             y = top_left[0]
#             similarity_value = min_val
#             print("INFO: min similarity value is {} in x = {} and y = {}.".format(min_val,x,y))

#         else:

#             top_left = max_loc
#             x = top_left[1]
#             y = top_left[0]
#             similarity_value = max_val
#             print("INFO: max similarity value is {} in x = {} and y = {}.".format(max_val,x,y))
#         #bottom_right = (top_left[0] + w, top_left[1] + h)

#         #TODO check if it is necessary

#         x = top_left[1]
#         y = top_left[0]

#         template_width, template_height = template.shape

#         if verbose > 1:
#             plt.imshow(roi)
#             rect = plt.Rectangle((y, x), template_height, template_width, color='red',
#                     fc='none')
#             plt.gca().add_patch(rect)
#             plt.title('Grayscale Image with Bounding Box around the pulse')
#             plt.show()

#         loc = np.where(res >= threshold)



#     if len(loc[0])>0:

#         detected = True # pulse was detected
#         ppts = np.array(list(map(list, zip(*loc[::-1])))) #obtain um array from the list of tuples
#         #print(ppts)
#         ppts_max = ppts[:,0].max()
#         ppts_min = ppts[:,0].min()
#         ppts_median = np.median(ppts[:,0])
#         #print(ppts_max, ppts_median, ppts_min)
#         extracted_pulse = roi[x:x+template_width, y:y+template_height]
#         # plt.imshow(extracted_pulse)
#         # plt.show()
#         _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
#         wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)

#     else:

#               # There was a pulse to be detected but the detection failed
#               # No pulse detected or the roi has no pulse
#               detected = False
#               #curve_scales.append((np.nan,np.nan))
#               wpulse = np.nan
#               hpulse = np.nan


#     return detected, location, similarity_value, x, y, wpulse, hpulse,



def print_segment_list(segment_list, plot_callback=None):
     for seg in segment_list:
          print("line number: {} - name: {} - segment length: {}".format (seg['line'],seg['name'] ,seg['lseg']))
          fig = plt.figure()
          plt.title(seg['name'])
          plt.plot(seg['xseg'], seg['yseg'])
          plt.grid()
          if plot_callback:
              plot_callback(plt.gcf())
              plt.close('all')
          else:
              plt.show()

def print_line_dict(line, plot_callback=None):
     for key, value in line.items():
      if key =='curves':
           print_segment_list(value, plot_callback=plot_callback)
      else:
          print(f"{key}: {value}")

def interpolate_segment(x, y, num):
     x_interp = np.linspace(0.0, 1.0, len(x))
     f = interpolate.CubicSpline(x_interp, y)
     x_new = np.linspace(0.0, 1.0, int(num))
     y_new = f(x_new)
     return x_new, y_new

def segment_to_df(line_list, pulse_per_sec, pulse_per_mv,num_pts):
    '''
    INPUT:
    line_list
    pulse_per_sec
    pulse_per_mv
    num_pts: number of points after the interpolation
    '''
    df = pd.DataFrame()

    #Check if at least one line 

    for line in line_list:
        for seg in (line['curves']):
            xsec, ymv= convert_to_secmv(seg['xseg'], seg['yseg'], line['wpulse'],
                                        line['hpulse'], seg['wseg'], seg['baseline'],
                                        pulse_per_sec, pulse_per_mv)
            x_new, y_new =  interpolate_segment(xsec, ymv, num_pts)
            df[seg['name']] = y_new
    return df


def process_line(line_number, labeled_line, offset, line_leads, config_dict, verbose=0):
    """ ECG line processing."""
    
    line_dict = {
        'wpulse': config_dict['wpulse'],
        'hpulse': config_dict['hpulse'],
        'curves': [],
        'offset_line': offset
    }

    if verbose > 1:
        display_segments(f"Labeled Line {line_number}", labeled_line,
                         plot_callback=config_dict.get('plot_callback'))

    u, c = np.unique(labeled_line, return_counts=True)
    # Sort labels by segment size descending (excluding background 0)
    segment_indices = np.argsort(-c[1:]) + 1
    app_seg_size = labeled_line.shape[1] // config_dict['layout'][1] # Calculates the approximate segment size 
    min_length_threshold = np.round(app_seg_size * 0.25, 0) #TODO: remove the hard code 0.25
    
    # Track label incrementing for split segments
    current_max_label = np.max(u)

    for label in segment_indices:
        slices = ndimage.find_objects(labeled_line == label) # return the slice that defines the bouding box of each element
        if not slices:
            continue
            
        sl_x, sl_y = slices[0]
        roi_length = sl_y.stop - sl_y.start
        
        if roi_length < min_length_threshold: # ignore small segments. It is probably garbage
            continue

        ratio = round(roi_length / app_seg_size, 0)
        
        # Skip if it's the rhythm lead or invalid ratio
        if line_number + 1 == config_dict['rhythm'] or ratio < 1.0:
            continue

        if verbose > 0:
            print(f"INFO: label = {label}, length = {roi_length}, ratio = {ratio}")

        # Number of segments to split this ROI into
        num_splits = int(min(ratio, config_dict['layout'][1])) # usually 4 , since the most common layout is  (3,4)
        split_width = (sl_y.stop - sl_y.start) // num_splits

        for i in range(num_splits):
            # Calculate dynamic Y boundaries
            start_y = sl_y.start + (i * split_width)
            # Ensure the last segment grabs the remainder of the pixels
            stop_y = sl_y.stop if i == num_splits - 1 else start_y + split_width
            
            # Create the segment label (original for first split, new labels for others)
            seg_label = label if i == 0 else (current_max_label + 1)
            if i > 0: current_max_label += 1

            # Extract and process the segment
            seg_roi = labeled_line[sl_x, start_y:stop_y]
            seg_binary = np.where(seg_roi == label, 255, 0).astype("uint8") #keep on the signal that corresponds to the label

            # Shared processing logic
            ws, ls, xs, ys = get_values_from_img(seg_binary)
            
            segment_dict = {
                'line': line_number,
                'label': seg_label,
                'start_x': sl_x.start,
                'stop_x': sl_x.stop,
                'start_y': start_y,
                'stop_y': stop_y,
                'wseg': ws,
                'lseg': ls,
                'xseg': xs,
                'yseg': ys,
                'baseline': np.argmax(np.std(seg_binary, axis=1))
            }

            # Specific logic for One Segment (Ratio 1.0)
            if ratio == 1.0:
                y_offset = np.argmax(seg_binary[:, 0]) if seg_binary.size > 0 else 0
                segment_dict['firstpixel_abs_y'] = y_offset + sl_x.start + ys[0]

            line_dict['curves'].append(segment_dict)
            
            if verbose > 0:
                print(f"INFO: Added segment label: {seg_label} length: {ls}")

    # Final sorting and lead naming
    line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])
    for i, curve in enumerate(line_dict['curves']):
        if i < len(line_leads):
            curve['name'] = line_leads[i]
            
    return line_dict
# def process_line(line_number, labeled_line, offset, line_leads, config_dict, verbose=0):
#     '''
#       process line of an ECG.
#       inputs: 
#                 line_number:
#                 labeled_line: the output of ndimage
#                 offset:  reference to locate the segment in the image
#                 line_leads: names of the leads in this line
#                 config_dict: configuration dictionary contains several info about the ECG, 
#                 such as layout, if it has calibration pulse
#                 verbose:
#     '''
#     # TODO: Clean this dictionary
#     line_dict ={}
#     line_dict['wpulse'] = config_dict['wpulse']
#     line_dict['hpulse'] = config_dict['hpulse']
#     line_dict['curves'] = []
#     line_dict['offset_line'] = offset


#     display_title = "Labeled Line" + str(line_number)
#     display_segments(display_title, labeled_line)

#     # if verbose > 1:
#     #      display_segments("Labeled Line", labeled_line)

#     u, c = np.unique(labeled_line, return_counts=True)
#     segment_labels = np.argsort(-c[1:]) +1 # sort label by segment size in decresent order
#     segment_length = -np.sort(-c[1:])
#     max_label = np.max(u)


#     app_seg_size = labeled_line.shape[1] // config_dict['layout'][1] # Calculates the approximate segment size 
#     if verbose > 1:
#         print("INFO: unique label {}.".format(u))
#         print("INFO: count {}.".format(c))
#         print("INFO: segment labels {}.".format(segment_labels))
#         print("INFO: segment lenghth {}.".format(segment_length))

#     larger_segments = segment_labels[:config_dict['layout'][1] + 1]

#     segment_ratios = []
   
#     temp = np.round(app_seg_size * 0.25, 0)
    
#     for l, label in enumerate(segment_labels):
        
#         #print ("INFO: label count  = {}.".format(c[label]))


#         roi = (labeled_line==label) # get the segment labeled with the label value
       
#         sl = ndimage.find_objects(roi) # return the slice that defines the bouding box of each element
#         if len(sl)==0:
#             continue
#         #print ("INFO: sl  = {}.".format(sl))
#         roi = roi[sl[0][0], sl[0][1]] # slice in x and slice in y
#         #print ("INFO: label count  = {}.".format(c[label]))
#         roi_length = roi.shape[1] # get the length of the segment
#         #print(roi_length)
#         if roi_length >= temp:
#             #roi_copy = (roi == label) * np.uint8(255) #np.where(roi == label, 255, 0).astype("uint8")
#             roi_copy = (roi) * np.uint8(255)

#             # calculate the ratio between length and approximate segment size
#             # to check if teh segmentation concatenate 2 ou more segments
#             ratio = round(roi_length / app_seg_size, 0) # calculate the ratio between length and appromate segment


#             if verbose > 0:
#                 print("INFO: label = {}, length = {} and ratio = {}.".format(label, roi_copy.shape[1], ratio))
#                 # plt.imshow(roi_copy)
#                 # plt.show()
#             # Calculate the ratio of the roi length in relation to the approximate segment size
#             # if the ratio == 4.0, we have four segments
#             # if the ratio == 3.0, we have three segments 
#             ratio = round(roi_length / app_seg_size, 0) 
#             segment_ratios.append(ratio)

#             if line_number + 1 == config_dict['rhythm']: # line number : 0,1,...
#             #discard rhythm
#                 pass
#             elif ratio == 4.0:
#                 print("INFO: Four Segments {}.".format(ratio))
#                 print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]))
#             elif ratio == 3.0:
#                 print("INFO: Three Segments {}.".format(ratio))
#                 print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]))

#                 # Separate the segments

#                 # First segment
#                 slx_seg1_start = sl[0][0].start
#                 slx_seg1_stop = sl[0][0].stop
#                 sly_seg1_start = sl[0][1].start
#                 sly_seg1_stop = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start) // 3)

#                 # append segment to the segment list
#                 segment_dict = {}
#                 segment_dict['line'] = line_number
#                 segment_dict['label'] = label
#                 segment_dict['start_x'] = slx_seg1_start
#                 segment_dict['stop_x'] = slx_seg1_stop
#                 segment_dict['start_y'] = sly_seg1_start
#                 segment_dict['stop_y'] = sly_seg1_stop

#                 #Take the slice from the labeled line
#                 seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
#                                slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

#                 seg = np.where(seg == label, 255, 0)
#                 #seg = seg.astype("uint8")

            
#                 ws, ls, xs, ys= get_values_from_img(seg)
#                 segment_dict['wseg'] = ws
#                 segment_dict['lseg'] = ls
#                 segment_dict['xseg'] = xs
#                 segment_dict['yseg'] = ys

#                 baseline = np.argmax(np.std(seg, axis=1))
#                 segment_dict['baseline'] = baseline

#                 line_dict['curves'].append(segment_dict)

#                 print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

#                 # Second Segment
#                 slx_seg2_start = sl[0][0].start
#                 slx_seg2_stop =  sl[0][0].stop
#                 sly_seg2_start = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start)//3)
#                 sly_seg2_stop = sl[0][1].start + ((sl[0][1].stop - sl[0][1].start)//3) + ((sl[0][1].stop -sl[0][1].start)//3)
#                 max_label = max_label+1 # add a new label

#                 # append segment to the segment list
#                 segment_dict = {}
#                 segment_dict['line'] = line_number
#                 segment_dict['label'] = max_label
#                 segment_dict['start_x'] = slx_seg2_start
#                 segment_dict['stop_x'] = slx_seg2_stop
#                 segment_dict['start_y'] = sly_seg2_start
#                 segment_dict['stop_y'] = sly_seg2_stop

#                 #Take the slice from the labeled line
#                 seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
#                                slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

#                 seg = np.where(seg == label, 255, 0)
#                 #seg = seg.astype("uint8")

#                 if verbose > 0 :
#                     title = "line: " + str(line_number) + "segment: " + str(label)
#                     plt.imshow(seg)
#                     plt.title(title)
#                     plt.show()

#                 ws, ls, xs, ys= get_values_from_img(seg)
#                 segment_dict['wseg'] = ws
#                 segment_dict['lseg'] = ls
#                 segment_dict['xseg'] = xs
#                 segment_dict['yseg'] = ys

#                 baseline = np.argmax(np.std(seg, axis =1))
#                 segment_dict['baseline'] = baseline

#                 line_dict['curves'].append(segment_dict)

#                 print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

#                 max_label = max_label + 1

#                 # Third Segment
#                 slx_seg3_start = sl[0][0].start
#                 slx_seg3_stop = sl[0][0].stop
#                 sly_seg3_start = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start) // 3) + ((sl[0][1].stop -sl[0][1].start)//3)
#                 sly_seg3_stop = sl[0][1].stop

#                 max_label = max_label   #add new label
#                 # append segment to the segment list
#                 segment_dict = {}
#                 segment_dict['line'] = line_number
#                 segment_dict['label'] = max_label
#                 segment_dict['start_x'] = slx_seg3_start
#                 segment_dict['stop_x'] = slx_seg3_stop
#                 segment_dict['start_y'] = sly_seg3_start
#                 segment_dict['stop_y'] = sly_seg3_stop

#                 #Take the slice from the labeled line
#                 seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
#                                 slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

#                 seg = np.where(seg==label,255,0)
#                 #seg = seg.astype("uint8")

#                 if verbose > 0 :
#                     title = "line: " + str(line_number) + "segment: " + str(label)
#                     plt.imshow(seg)
#                     plt.title(title)
#                     plt.show()

#                 # get the x,y values from the image
#                 ws, ls, xs, ys= get_values_from_img(seg)
#                 segment_dict['wseg'] = ws
#                 segment_dict['lseg'] = ls
#                 segment_dict['xseg'] = xs
#                 segment_dict['yseg'] = ys


#                 baseline = np.argmax(np.std(seg, axis=1))
#                 segment_dict['baseline'] = baseline

#                 line_dict['curves'].append(segment_dict)

#                 print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

#             elif ratio == 2.0:
#                 print("INFO: two  segments {} Slice X = {} and Slice Y =  {}" .format(ratio, sl[0][0], sl[0][1]) )

#                 # Separate the segments

#                 # First segment
#                 slx_seg1_start = sl[0][0].start
#                 slx_seg1_stop = sl[0][0].stop
#                 sly_seg1_start = sl[0][1].start
#                 sly_seg1_stop = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start)//2)

#                 # append segment to the segment list
#                 segment_dict = {}
#                 segment_dict['line'] = line_number
#                 segment_dict['label'] = label
#                 segment_dict['start_x'] = slx_seg1_start
#                 segment_dict['stop_x'] = slx_seg1_stop
#                 segment_dict['start_y'] = sly_seg1_start
#                 segment_dict['stop_y'] = sly_seg1_stop

#                 #Take the slice from the labeled line
#                 seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
#                                 slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

#                 seg = np.where(seg == label, 255, 0)
#                 #seg = seg.astype("uint8")

#                 if verbose > 0 :
#                     title = "line: " + str(line_number) + " segment: " + str(label)
#                     plt.imshow(seg)
#                     plt.title(title)
#                     plt.show()

#                 ws, ls, xs, ys= get_values_from_img(seg)
#                 segment_dict['wseg'] = ws
#                 segment_dict['lseg'] = ls
#                 segment_dict['xseg'] = xs
#                 segment_dict['yseg'] = ys

#                 baseline = np.argmax(np.std(seg, axis=1))
#                 segment_dict['baseline'] = baseline

#                 line_dict['curves'].append(segment_dict)

#                 print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

#                 # Second segments
#                 slx_seg2_start = sl[0][0].start
#                 slx_seg2_stop = sl[0][0].stop
#                 sly_seg2_start = sl[0][1].start + ((sl[0][1].stop - sl[0][1].start)//2)
#                 sly_seg2_stop = sl[0][1].stop

#                 max_label = max_label + 1

#                 # append segment to the segment list
#                 segment_dict = {}
#                 segment_dict['line'] = line_number
#                 segment_dict['label'] = max_label
#                 segment_dict['start_x'] = slx_seg2_start
#                 segment_dict['stop_x'] = slx_seg2_stop
#                 segment_dict['start_y'] = sly_seg2_start
#                 segment_dict['stop_y'] = sly_seg2_stop

#                 #Take the slice from the labeled line
#                 seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
#                                 slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

#                 seg = np.where(seg == label, 255, 0)
#                 # seg = seg.astype("uint8")

#                 if verbose > 0 :
#                     title = "line: "+ str(line_number) + " segment : " + str(label)
#                     plt.imshow(seg)
#                     plt.title(title)
#                     plt.show()

#                 ws, ls, xs, ys= get_values_from_img(seg)
#                 segment_dict['wseg'] = ws
#                 segment_dict['lseg'] = ls
#                 segment_dict['xseg'] = xs
#                 segment_dict['yseg'] = ys

#                 baseline = np.argmax(np.std(seg, axis =1))
#                 segment_dict['baseline'] = baseline

#                 line_dict['curves'].append(segment_dict)

#                 print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

#             elif ratio == 1.0:
#                 print("INFO: One Segment {}.".format(ratio))
#                 print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

#                 slx_seg1_start = sl[0][0].start
#                 slx_seg1_stop = sl[0][0].stop
#                 sly_seg1_start = sl[0][1].start
#                 sly_seg1_stop = sl[0][1].stop


#                 # append segment to the segment list

#                 #segment_dict = fill_slice_info(line_number,label, slx_seg1_start, slx_seg1_stop,sly_seg1_start, sly_seg1_stop)
#                 segment_dict = {}
#                 segment_dict['line'] = line_number
#                 segment_dict['label'] = label
#                 segment_dict['start_x'] = slx_seg1_start
#                 segment_dict['stop_x'] = slx_seg1_stop
#                 segment_dict['start_y'] = sly_seg1_start
#                 segment_dict['stop_y'] = sly_seg1_stop

#                 #Take the slice from the labeled line
#                 seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
#                                 slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

#                 seg = np.where(seg == label, 255, 0)

#                 if verbose > 0:
#                     title = "line: " + str(line_number) + " segment: " + str(label)
#                     plt.imshow(seg)
#                     plt.title(title)
#                     plt.show()

#                 ws, ls, xs, ys= get_values_from_img(seg)
#                 segment_dict['wseg'] = ws
#                 segment_dict['lseg'] = ls
#                 segment_dict['xseg'] = xs
#                 segment_dict['yseg'] = ys
                
#                 segment_dict['firstpixel_abs_y'] = np.argmax(seg[:,0]) + segment_dict['start_x'] + ys[0]

#                 baseline = np.argmax(np.std(seg, axis =1))
#                 segment_dict['baseline'] = baseline

#                 print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

#                 line_dict['curves'].append(segment_dict)

#             elif ratio < 1.0:
#                     dummy = 0 
#                     #print("INFO: Garbage {}.".format(ratio))
#                     continue
#         else:
#             dummy = 0
#             #print("INFO: Garbage length {} label  {} {}.".format(roi_length, label, l))
            
#     line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])
#     for i, d in enumerate(line_dict['curves']):
#         d['name'] = line_leads[i]  #add  the name of the segments
#     return line_dict

# def plot_ecg(df,columns,title, n_rows = 4, n_columns = 4, x_spacing = 100, y_spacing = 0.1, figure_size = (20, 12)):
#     if (n_rows * n_columns) < len(columns):
#         raise Exception('Columns must be the equal or smaller than the number of rows and columns.')
#     fig, axes = plt.subplots(n_rows, n_columns, figsize = figure_size)
#     fig.suptitle(title,fontsize = 20)
    
#     for index,col in enumerate(columns):
        
#         if n_rows == 1 or n_columns == 1:
#             current_ax = axes[index]
#         else:
#             row_index = int(index/n_columns)
#             col_index = int(index - n_columns*row_index)
#             ax = axes[row_index][col_index]
        
        
#         ax.plot(df[col]) 
#         ax.set_title(col)
#         #y_ticks = np.linspace(df[col].min(),df[col].max(),10)
#         y_ticks = np.arange(df[col].min(),df[col].max(),y_spacing)
#         ax.set_yticks(y_ticks)
#         x_max = len(df[col].values)
#         #x_ticks = np.linspace(0,x_max,10, endpoint= False)
#         x_ticks =list(range(0,x_max,x_spacing))
#         ax.tick_params(axis='x', rotation=90)
#         ax.set_xticks(x_ticks)
#         #label = r'$\mu={:2.2f},\ \sigma={:2.2f},\ median={:2.2f},\ mode={:2.2f}$'.format(df[col].mean(),df[col].std(),df[col].median(),df[col].mode().values[0])
#         #ax.set_xlabel(label)
#         ax.grid(True)
#     plt.subplots_adjust(top=0.92,hspace = 0.45,wspace = 0.5)    
#     plt.show()

def plot_ecg(df,columns,title, n_rows = 4, n_columns = 4, fs = 500, figure_size = (20, 12)):
    if (n_rows * n_columns) < len(columns):
        raise Exception('Columns must be the equal or smaller than the number of rows and columns.')
    fig, axes = plt.subplots(n_rows, n_columns, figsize = figure_size)
    fig.suptitle(title,fontsize = 20)
    
    for index,col in enumerate(columns):
        
        if n_rows == 1 or n_columns == 1:
            current_ax = axes[index]
        else:
            row_index = int(index/n_columns)
            col_index = int(index - n_columns*row_index)
            ax = axes[row_index][col_index]
            
        
        
        # ax.plot(df[col]) 
        # ax.set_title(col)
        signal = df[col]
        ts = np.arange(signal.size) / fs

        ax= plot_ecg_signal(ts, signal, ax)
        
    plt.subplots_adjust(top=0.92,hspace = 0.45,wspace = 0.5)    
    plt.show()
    return fig

def plot_ecg_signal(time, signal,ax):
    #fig = plt.figure(figsize=(15, 3));
    # ax = plt.axes();
    ax.plot(time, signal);
    # setup major and minor ticks
    min_t = int(np.min(time))
    max_t = round(np.max(time))
    major_ticks = np.arange(min_t, max_t+1)
    ax.set_xticks(major_ticks)
    # Turn on the minor ticks on
    ax.minorticks_on()
    # Make the major grid
    ax.grid(which='major', linestyle='-', color='red', linewidth='1.0')
    # Make the minor grid
    ax.grid(which='minor', linestyle=':', color='black', linewidth='0.5')
    plt.xlabel('Time (sec)');
    plt.ylabel('Amplitude')
    return ax
