
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
    vr, hr = A.shape[0] // blockdims[0], A.shape[1] // blockdims[1]
    B = np.zeros((vr,hr))

    verts = np.vsplit(A, vr)
    for i in range(len(verts)):
       for j, v in enumerate(np.hsplit(verts[i], hr)):
          B[i,j]=(np.std(A[
             i * blockdims[0] : (i + 1) * blockdims[0],
             j * blockdims[1] : (j + 1) * blockdims[1]
            ]))
    return B

def display_segments(name, item, axis='off'):
    plt.figure(figsize=(12, 9))
    plt.imshow(item, cmap="magma")
    plt.title(name)
    plt.axis(axis)
    plt.subplots_adjust(wspace=0.05, left=0.01, bottom=0.01, right=0.99, top=0.9)
    plt.show()

def get_values_from_img(roi):
    '''
    get the values of coord x and y for the image that contain the signal
    INPUT:
        roi: binary image with signal in white
    OUTPUT:
        xs, ys: values of the signal
    '''
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]
    width, length = roi.shape[:2]
    xs, ys = [], []
    bool_roi = roi != 0
    old_dy_dx = 0.0
    for i, col in enumerate(bool_roi.T):
        if len(xs) != 0:
            label, num = ndimage.label(col, structure=np.ones((3,)))
            if num != 0:
                median_list = []
                dy_dx_list = []
                d2y_dx2_list = []
                for n in range(1, num + 1):
                    pixel_loc = width - np.where(label == n)[0]
                    median_pixel = find_nearest(pixel_loc, np.median(pixel_loc))
                    median_list.append(median_pixel)
                    dy_dx = (median_pixel - ys[-1]) #/ (i - xs[-1])
                    d2y_dx2 = (dy_dx - old_dy_dx) #/ (i - xs[-1])
                    dy_dx_list.append(dy_dx)
                    d2y_dx2_list.append(d2y_dx2)
                tmp = np.argmin(np.abs(d2y_dx2_list))
                old_dy_dx = dy_dx_list[tmp]
                xs.append(i)
                ys.append(median_list[tmp])
        else:
            pixel_loc = width - np.where(col)[0]
            if pixel_loc.size > 0:
                median_pixel = np.median(pixel_loc)
                xs.append(i)
                ys.append(median_pixel)
    return width, length, xs, ys

def measure_extract_pulse(x, y, verbose=0):
    min_pulse = np.min(y)
    max_pulse = np.max(y)

    height = np.max(max_pulse-min_pulse)
    threshold = height / 2
    index = np.where((y - min_pulse)>=threshold)[0]
    width = x[index[-1]] - x[index[0]]
    if verbose > 0:
        print(f"pulse height: {height}")
        print(f"pulse width: {width} time units")
    return width, height

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

def detect_ref_pulse(roi, template,location='right', threshold=0.6, verbose=2):
    '''
    '''
    if roi.shape[0] <= template.shape[0] or roi.shape[1] <= template.shape[1]:
        #template is bigger than roi. Can not perform matchTemplate

        empty_list=[]
        empty_array = np.array(empty_list)
        loc = (empty_array,empty_array )
    else:
        method = cv.TM_CCORR_NORMED
        res = cv.matchTemplate(roi,template,method) # try tofind the pulse using a template match
        # Getting the max
        # x, y = np.unravel_index(np.argmax(res), res.shape)
        # print("INFO: max correlation is {} in x = {} and y = {}.".format(np.max(res),x,y))

        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
            x = top_left[1]
            y = top_left[0]
            similarity_value = min_val
            print("INFO: min similarity value is {} in x = {} and y = {}.".format(min_val,x,y))
        else:
            top_left = max_loc
            x = top_left[1]
            y = top_left[0]
            similarity_value = max_val
            print("INFO: max similarity value is {} in x = {} and y = {}.".format(max_val,x,y))
        #bottom_right = (top_left[0] + w, top_left[1] + h)

        #TODO check if it is necessary
        x = top_left[1]
        y = top_left[0]

        template_width, template_height = template.shape
        if verbose > 1:
            plt.imshow(roi)
            rect = plt.Rectangle((y, x), template_height, template_width, color='red',
                    fc='none')
            plt.gca().add_patch(rect)
            plt.title('Grayscale Image with Bounding Box around the pulse')
            plt.show()

        loc = np.where(res >= threshold)

    if len(loc[0])>0:
        detected = True # pulse was detected

        ppts = np.array(list(map(list, zip(*loc[::-1])))) #obtain um array from the list of tuples
        #print(ppts)
        ppts_max = ppts[:,0].max()
        ppts_min = ppts[:,0].min()
        ppts_median = np.median(ppts[:,0])
        #print(ppts_max, ppts_median, ppts_min)

        extracted_pulse = roi[x:x+template_width, y:y+template_height]
        # plt.imshow(extracted_pulse)
        # plt.show()
        _,_,xpulse,ypulse= get_values_from_img(extracted_pulse)
        wpulse,hpulse = measure_extract_pulse(xpulse,ypulse)

    else:
              # There was a pulse to be detected but the detection failed
              # No pulse detected or the roi has no pulse
              detected = False
              #curve_scales.append((np.nan,np.nan))
              wpulse = np.nan
              hpulse = np.nan

    return detected, location, similarity_value, x, y, wpulse, hpulse,

def print_segment_list(segment_list):
     for seg in segment_list:
          print("line number: {} - name: {} - segment length: {}".format (seg['line'],seg['name'] ,seg['lseg']))
          fig = plt.figure()
          plt.title(seg['name'])
          plt.plot(seg['xseg'], seg['yseg'])
          plt.grid()
          plt.show()

def print_line_dict(line):
     for key, value in line.items():
      if key =='curves':
           print_segment_list(value)
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

def remove_text(image, confidence_threshold):
    image_copy = image.copy()
    results = pytesseract.image_to_data(image_copy, config='--psm 11', output_type='dict')

    for i in range(len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # Extract the confidence of the text
        conf = int(results["conf"][i])

        if conf > 100 * confidence_threshold: # adjust to your liking
            # Cover the text with a black rectangle
            print("INFO: word detect in the image")
            cv.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 0), -1)
    return image_copy

def process_line(line_number, labeled_line, offset, line_leads, config_dict, verbose=0):
    '''
    '''
    # TODO: Clean this dictionary
    line_dict ={}
    line_dict['wpulse'] = config_dict['wpulse']
    line_dict['hpulse'] = config_dict['hpulse']
    line_dict['curves'] = []
    line_dict['offset_line'] = offset


    display_title = "Labeled Line" + str(line_number)
    display_segments(display_title, labeled_line)

    # if verbose > 1:
    #      display_segments("Labeled Line", labeled_line)

    u, c = np.unique(labeled_line, return_counts=True)
    segment_labels = np.argsort(-c[1:]) +1 # sort label by segment size in decresent order
    segment_length = -np.sort(-c[1:])
    max_label = np.max(u)


    app_seg_size = labeled_line.shape[1] // config_dict['layout'][1]
    if verbose > 1:
        print("INFO: unique label {}.".format(u))
        print("INFO: count {}.".format(c))
        print("INFO: segment labels {}.".format(segment_labels))
        print("INFO: segment lenghth {}.".format(segment_length))

    larger_segments = segment_labels[:config_dict['layout'][1] + 1]

    segment_ratios = []
   
    temp = np.round(app_seg_size * 0.25, 0)
    
    for l, label in enumerate(segment_labels):
        
        #print ("INFO: label count  = {}.".format(c[label]))


        roi = (labeled_line==label)
       
        sl = ndimage.find_objects(roi.astype(np.int32))
        if len(sl)==0:
            continue
        #print ("INFO: sl  = {}.".format(sl))
        roi = roi[sl[0][0], sl[0][1]] # slice in x and slice in y
        #print ("INFO: label count  = {}.".format(c[label]))
        roi_length = roi.shape[1]
        #print(roi_length)
        if roi_length >= temp:
            #roi_copy = (roi == label) * np.uint8(255) #np.where(roi == label, 255, 0).astype("uint8")
            roi_copy = (roi) * np.uint8(255)

            # calculate the ratio between length and approximate segment size
            # to check if teh segmentation concatenate 2 ou more segments
            ratio = round(roi_length / app_seg_size, 0) # calculate the ratio between length and appromate segment


            if verbose > 0:
                print("INFO: label = {}, length = {} and ratio = {}.".format(label, roi_copy.shape[1], ratio))
                # plt.imshow(roi_copy)
                # plt.show()

            ratio = round(roi_length / app_seg_size, 0)
            segment_ratios.append(ratio)

            if line_number + 1 == config_dict['rhythm']: # line number : 0,1,...
            #discard rhythm
                pass
            elif ratio == 4.0:
                print("INFO: Four Segments {}.".format(ratio))
                print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]))
            elif ratio == 3.0:
                print("INFO: Three Segments {}.".format(ratio))
                print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]))

                # Separate the segments

                # First segment
                slx_seg1_start = sl[0][0].start
                slx_seg1_stop = sl[0][0].stop
                sly_seg1_start = sl[0][1].start
                sly_seg1_stop = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start) // 3)

                # append segment to the segment list
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] = label
                segment_dict['start_x'] = slx_seg1_start
                segment_dict['stop_x'] = slx_seg1_stop
                segment_dict['start_y'] = sly_seg1_start
                segment_dict['stop_y'] = sly_seg1_stop

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
                               slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

                seg = np.where(seg == label, 255, 0)
                #seg = seg.astype("uint8")

            
                ws, ls, xs, ys= get_values_from_img(seg)
                segment_dict['wseg'] = ws
                segment_dict['lseg'] = ls
                segment_dict['xseg'] = xs
                segment_dict['yseg'] = ys

                baseline = np.argmax(np.std(seg, axis=1))
                segment_dict['baseline'] = baseline

                line_dict['curves'].append(segment_dict)

                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

                # Second Segment
                slx_seg2_start = sl[0][0].start
                slx_seg2_stop =  sl[0][0].stop
                sly_seg2_start = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start)//3)
                sly_seg2_stop = sl[0][1].start + ((sl[0][1].stop - sl[0][1].start)//3) + ((sl[0][1].stop -sl[0][1].start)//3)
                max_label = max_label+1 # add a new label

                # append segment to the segment list
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] = max_label
                segment_dict['start_x'] = slx_seg2_start
                segment_dict['stop_x'] = slx_seg2_stop
                segment_dict['start_y'] = sly_seg2_start
                segment_dict['stop_y'] = sly_seg2_stop

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
                               slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

                seg = np.where(seg == label, 255, 0)
                #seg = seg.astype("uint8")

                if verbose > 0 :
                    title = "line: " + str(line_number) + "segment: " + str(label)
                    plt.imshow(seg)
                    plt.title(title)
                    plt.show()

                ws, ls, xs, ys= get_values_from_img(seg)
                segment_dict['wseg'] = ws
                segment_dict['lseg'] = ls
                segment_dict['xseg'] = xs
                segment_dict['yseg'] = ys

                baseline = np.argmax(np.std(seg, axis =1))
                segment_dict['baseline'] = baseline

                line_dict['curves'].append(segment_dict)

                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

                max_label = max_label + 1

                # Third Segment
                slx_seg3_start = sl[0][0].start
                slx_seg3_stop = sl[0][0].stop
                sly_seg3_start = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start) // 3) + ((sl[0][1].stop -sl[0][1].start)//3)
                sly_seg3_stop = sl[0][1].stop

                max_label = max_label   #add new label
                # append segment to the segment list
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] = max_label
                segment_dict['start_x'] = slx_seg3_start
                segment_dict['stop_x'] = slx_seg3_stop
                segment_dict['start_y'] = sly_seg3_start
                segment_dict['stop_y'] = sly_seg3_stop

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
                                slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

                seg = np.where(seg==label,255,0)
                #seg = seg.astype("uint8")

                if verbose > 0 :
                    title = "line: " + str(line_number) + "segment: " + str(label)
                    plt.imshow(seg)
                    plt.title(title)
                    plt.show()

                # get the x,y values from the image
                ws, ls, xs, ys= get_values_from_img(seg)
                segment_dict['wseg'] = ws
                segment_dict['lseg'] = ls
                segment_dict['xseg'] = xs
                segment_dict['yseg'] = ys


                baseline = np.argmax(np.std(seg, axis=1))
                segment_dict['baseline'] = baseline

                line_dict['curves'].append(segment_dict)

                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))
            elif ratio == 2.0:
                print("INFO: two  segments {} Slice X = {} and Slice Y =  {}" .format(ratio, sl[0][0], sl[0][1]) )

                # Separate the segments

                # First segment
                slx_seg1_start = sl[0][0].start
                slx_seg1_stop = sl[0][0].stop
                sly_seg1_start = sl[0][1].start
                sly_seg1_stop = sl[0][1].start + ((sl[0][1].stop -sl[0][1].start)//2)

                # append segment to the segment list
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] = label
                segment_dict['start_x'] = slx_seg1_start
                segment_dict['stop_x'] = slx_seg1_stop
                segment_dict['start_y'] = sly_seg1_start
                segment_dict['stop_y'] = sly_seg1_stop

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
                                slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

                seg = np.where(seg == label, 255, 0)
                #seg = seg.astype("uint8")

                if verbose > 0 :
                    title = "line: " + str(line_number) + " segment: " + str(label)
                    plt.imshow(seg)
                    plt.title(title)
                    plt.show()

                ws, ls, xs, ys= get_values_from_img(seg)
                segment_dict['wseg'] = ws
                segment_dict['lseg'] = ls
                segment_dict['xseg'] = xs
                segment_dict['yseg'] = ys

                baseline = np.argmax(np.std(seg, axis=1))
                segment_dict['baseline'] = baseline

                line_dict['curves'].append(segment_dict)

                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

                # Second segments
                slx_seg2_start = sl[0][0].start
                slx_seg2_stop = sl[0][0].stop
                sly_seg2_start = sl[0][1].start + ((sl[0][1].stop - sl[0][1].start)//2)
                sly_seg2_stop = sl[0][1].stop

                max_label = max_label + 1

                # append segment to the segment list
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] = max_label
                segment_dict['start_x'] = slx_seg2_start
                segment_dict['stop_x'] = slx_seg2_stop
                segment_dict['start_y'] = sly_seg2_start
                segment_dict['stop_y'] = sly_seg2_stop

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
                                slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

                seg = np.where(seg == label, 255, 0)
                # seg = seg.astype("uint8")

                if verbose > 0 :
                    title = "line: "+ str(line_number) + " segment : " + str(label)
                    plt.imshow(seg)
                    plt.title(title)
                    plt.show()

                ws, ls, xs, ys= get_values_from_img(seg)
                segment_dict['wseg'] = ws
                segment_dict['lseg'] = ls
                segment_dict['xseg'] = xs
                segment_dict['yseg'] = ys

                baseline = np.argmax(np.std(seg, axis =1))
                segment_dict['baseline'] = baseline

                line_dict['curves'].append(segment_dict)

                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

            elif ratio == 1.0:
                print("INFO: One Segment {}.".format(ratio))
                print("INFO:Slice X = {} and Slice Y =  {}" .format(sl[0][0], sl[0][1]) )

                slx_seg1_start = sl[0][0].start
                slx_seg1_stop = sl[0][0].stop
                sly_seg1_start = sl[0][1].start
                sly_seg1_stop = sl[0][1].stop


                # append segment to the segment list

                #segment_dict = fill_slice_info(line_number,label, slx_seg1_start, slx_seg1_stop,sly_seg1_start, sly_seg1_stop)
                segment_dict = {}
                segment_dict['line'] = line_number
                segment_dict['label'] = label
                segment_dict['start_x'] = slx_seg1_start
                segment_dict['stop_x'] = slx_seg1_stop
                segment_dict['start_y'] = sly_seg1_start
                segment_dict['stop_y'] = sly_seg1_stop

                #Take the slice from the labeled line
                seg = labeled_line[slice(*(segment_dict['start_x'], segment_dict['stop_x'], None)),
                                slice(*(segment_dict['start_y'], segment_dict['stop_y'], None))]

                seg = np.where(seg == label, 255, 0)

                if verbose > 0:
                    title = "line: " + str(line_number) + " segment: " + str(label)
                    plt.imshow(seg)
                    plt.title(title)
                    plt.show()

                ws, ls, xs, ys= get_values_from_img(seg)
                segment_dict['wseg'] = ws
                segment_dict['lseg'] = ls
                segment_dict['xseg'] = xs
                segment_dict['yseg'] = ys
                
                segment_dict['firstpixel_abs_y'] = np.argmax(seg[:,0]) + segment_dict['start_x'] + ys[0]

                baseline = np.argmax(np.std(seg, axis =1))
                segment_dict['baseline'] = baseline

                print("INFO: label: {}  length {}".format(segment_dict['label'], segment_dict['lseg']))

                line_dict['curves'].append(segment_dict)

            elif ratio < 1.0:
                    dummy = 0 
                    #print("INFO: Garbage {}.".format(ratio))
                    continue
        else:
            dummy = 0
            #print("INFO: Garbage length {} label  {} {}.".format(roi_length, label, l))
            
    line_dict['curves'] = sorted(line_dict['curves'], key=lambda d: d['start_y'])
    for i, d in enumerate(line_dict['curves']):
        d['name'] = line_leads[i]  #add  the name of the segments
    return line_dict

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
