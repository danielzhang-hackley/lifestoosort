import cv2
import numpy as np


def move_output_bin(kit, type):
	if (type == "screw"):
		kit.servo[1].angle = 40
	elif (type == "bolt"):
		kit.servo[1].angle = 140
	else:
		kit.servo[1].angle = 90



# short side no lines, long side parallel to concave = screw
# short side no lines, long side perpendicular to concave = nut
# short side with lines, long slide parallel to concave = bolt

def screw_bolt_other(image, blur=(5,5), light_threshold=148, thresholding=cv2.THRESH_BINARY,
                     canny_t_lower=300, canny_t_upper=400):
    output = [None for _ in range(5)]  # type, reasoning, drawings, thresh, head

    # PROCESS IMAGE AND CREATE CONTOURS
    img = image.copy()  # keep the original image clean
    grayscale = cv2.cvtColor(cv2.blur(img, blur), cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(grayscale, light_threshold, 255, thresholding)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output[2], output[3] = img, thresh


    if len(contours) == 0:
        return tuple(output)

    # FIND LARGEST AREA CONTOUR
    cnt = contours[0]
    max_area = cv2.contourArea(cnt)
    for cont in contours:
        if cv2.contourArea(cont) > max_area:
            cnt = cont
            max_area = cv2.contourArea(cont)


    img, cnt = straighten_image(img, thresh, cnt)

    epsilon = 0.005*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # CREATE CONVEX HULL
    convex_hull_points = cv2.convexHull(approx)
    convex_hull = cv2.convexHull(approx, returnPoints=False)

    # draw poly estimate and convex hull
    cv2.drawContours(img, [approx], -1, (0, 0, 255), 3)
    cv2.drawContours(img, [convex_hull_points], -1, (255, 0, 0), 2)

    try:
        convexity_defects = cv2.convexityDefects(approx, convex_hull)
    except:
        output[0] = "other"
        return tuple(output)

    if convexity_defects is None:
        output[0] = "other"
        return tuple(output)

    # FIND CONVEXITY DEFECT LOCATIONS
    # find the indices of convexity defect where the distance between hull and approximation is greatest
    p1_idx = [0, 0]  # the index in convexity_defects of the max, max
    p2_idx = [0, 0]
    for i, defect in enumerate(convexity_defects):
        defect = defect[0]
        if defect[3] > p1_idx[1]:
            p2_idx = p1_idx
            p1_idx = [i, defect[3]]
        elif defect[3] > p2_idx[1]:
            p2_idx = [i, defect[3]]

    # find the indices of the second closest pair of points on the convex hull
    closest = [(0, -1), (0, -1), float('inf')]   # (1st index in approx which is element of convexity_defects, upper or lower),
                                                 # (2nd index, upper lower), min
    second_closest = [(0, -1), (0, -1), float('inf')]
    for i, first_point_idx in enumerate(convexity_defects[p1_idx[0]][0][0:2]):
        for j, second_point_idx in enumerate(convexity_defects[p2_idx[0]][0][0:2]):
            first_point = approx[first_point_idx][0]
            second_point = approx[second_point_idx][0]

            difference = second_point - first_point
            distance = np.linalg.norm(difference)

            if distance < closest[-1]:
                second_closest = closest
                closest = [(first_point_idx, i), (second_point_idx, j), distance]
            elif distance < second_closest[-1]:
                second_closest = [(first_point_idx, i), (second_point_idx, j), distance]

    # FIND NOTCHES
    # if lower bound, we can move forward one, if upper bound, we can move backwards one
    # dct = {0: 1, 1: -1}
    notch1_idx = second_closest[0][0] # + dct[second_closest[0][1]]
    notch2_idx = second_closest[1][0] # + dct[second_closest[1][1]]
    notch1 = approx[notch1_idx][0]
    notch2 = approx[notch2_idx][0]

    # FIND BOUNDING BOX FOR HEAD
    '''
    # split the whole box at the points of concavity first
    """
    p1, p2, p3, p4 = whole_box[0], whole_box[1], whole_box[2], whole_box[3]
    l1 = mat_line(p1, p2)
    l2 = mat_line(p2, p3)
    """
    # concave_line_eq = mat_line(notch1, notch2)
    """
    # if the the line without an intersection is longer than the one with, then it's
    # neither a bolt nor a screw
    # the line with an intersection corresponds to the side of the head, not the top
    l1_concave_inter = find_intersect(l1, concave_line)
    l1_concave_inter_valid = intersection_in_range(l1_concave_inter, p1, p2)
    l2_concave_inter = find_intersect(l2, concave_line)
    l2_concave_inter_valid = intersection_in_range(l2_concave_inter, p2, p3)
    if (not (l1_concave_inter_valid or l2_concave_inter_valid)) or (l1_concave_inter_valid and l2_concave_inter_valid):
        output[0] = "other"
        return output
    else:
        l1_length = np.linalg.norm([p1, p2])
        l2_length = np.linalg.norm([p2, p3])
        if (l2_length < l1_length and l2_concave_inter_valid) or (l1_length < l2_length and l1_concave_inter_valid):
            output[0] = "other"
            output[1] = ("concave on short side")
    if l1_concave_inter_valid:
        l3 = mat_line(p3, p4)
        l3_concave_inter = find_intersect(l3, concave_line)
        whole_box_sect_1 = np.array([p1, l1_concave_inter, l3_concave_inter, p4])
        whole_box_sect_2 = np.array([l1_concave_inter, p2, p3, l3_concave_inter])
    else:
        l3 = mat_line(p1, p4)
        l3_concave_inter = find_intersect(l3, concave_line)
        whole_box_sect_1 = np.array([p1, p2, l2_concave_inter, l3_concave_inter])
        whole_box_sect_2 = np.array([l2_concave_inter, p3, p4, l3_concave_inter])
    """
    base1 = (whole_box_points[1] - whole_box_points[0]) / np.linalg.norm(whole_box_points[1] - whole_box_points[0])
    base2 = (whole_box_points[2] - whole_box_points[1]) / np.linalg.norm(whole_box_points[2] - whole_box_points[1])
    approx_s1 = {}
    approx_s2 = {}
    approx_s1["shape"] = np.vstack([approx[: min(notch1_idx, notch2_idx) + 1],
                                 approx[max(notch1_idx, notch2_idx): ]])
    approx_s2["shape"] = approx[min(notch1_idx, notch2_idx): max(notch1_idx, notch2_idx) + 1]
    approx_s1["box"] = cv2.minAreaRect(approx_s1["shape"])
    approx_s2["box"] = cv2.minAreaRect(approx_s2["shape"])
    approx_s1["count"], approx_s1["edges"], approx_s1["eqs"], approx_s1["transformation"] = \
        count_parallel_lines(image, approx_s1["box"], t_lower=canny_t_lower, t_upper=canny_t_upper)
    approx_s2["count"], approx_s2["edges"], approx_s2["eqs"], approx_s2["transformation"] = \
        count_parallel_lines(image, approx_s2["box"], t_lower=canny_t_lower, t_upper=canny_t_upper)
    if approx_s1["count"] == 0 and approx_s2["count"] == 0:
        output[0] = "other"
        return output
    approx_s1["concave eq"] = mat_line()
    approx_s1["crosses concave"] = False
    approx_s2["crosses concave"] = False
    # print(approx_s1["count"])
    # print(approx_s2["count"])
    for approx_s in (approx_s1, approx_s2):
        for eq in approx_s["eqs"]:
            if intersection_in_range_2(eq, approx_s["concave eq"], notch1, notch2):
                print("true")
                approx_s["crosses concave"] = True
    cv2.imshow('edges1', approx_s1["edges"])
    cv2.imshow('edges2', approx_s2["edges"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # a screw has lines that don't intersect in shaft, and no lines at all in head    | lines > 7 that dont intersect, no lines
    # a bolt has lines that don't intersect in shaft, and lines that intersect in head | lines > 7 that dont intersect, lines > 1 that intersect
    # certain nuts have no lines at all in shaft, and lines that intersect in head    | no lines, lines > 1 that intersect
    # screw: perp edge, no lines
    # bolt: perp edge, perp edge
    # nut: perp edge, no lines
    n_have_lines = int(approx_s1["count"] >= 2) + int(approx_s2["count"] >= 2)  # number of boxes that have sufficient lines
    if n_have_lines == 0:
        output[0] = "other"
        return output
    elif n_have_lines == 1:
        print(approx_s1["count"])
        print(approx_s2["count"])
        print(approx_s1["crosses concave"])
        print(approx_s2["crosses concave"])
        print(len(approx_s1["eqs"]))
        print(len(approx_s2["eqs"]))
        approx_with_lines = approx_s1 if approx_s1["count"] >= 2 else approx_s2
        if approx_with_lines["crosses concave"]:
            output[0] = "other"
            return output
        else:
            output[0] = "screw"
            return output
    else:
        output[0] = "bolt"
        return output
    '''

    concave_line_eq = mat_line(notch1, notch2)

    if abs(np.dot(concave_line_eq[0], np.array([0, 1]))) < 0.1:
        output[0] = "other"
        return output[0]

    approx_s1 = np.vstack([approx[: min(notch1_idx, notch2_idx) + 1],
                             approx[max(notch1_idx, notch2_idx): ]])
    approx_s2 = approx[min(notch1_idx, notch2_idx): max(notch1_idx, notch2_idx) + 1]
    # approx_head = approx[min(notch1_idx, notch2_idx): max(notch1_idx, notch2_idx) + 1]
    box1 = cv2.minAreaRect(approx_s1)
    box2 = cv2.minAreaRect(approx_s2)

    if box1[0][1] < box2[0][1]:
        head_box = box1
        shaft_box = box2
    else:
        head_box = box2
        shaft_box = box1


    # DRAW EVERYTHING ELSE
    head_box_points = np.int0(cv2.boxPoints(head_box))
    img = cv2.circle(img, notch1, radius=3, color=(0, 255, 0), thickness=-1)
    img = cv2.circle(img, notch2, radius=3, color=(0, 255, 0), thickness=-1)
    cv2.drawContours(img, [head_box_points], 0, (255, 0, 255), 2)

    output[2] = img

    # cv2.drawContours(img,[whole_box], 0, (255, 255, 0), 2)

    '''
    # FIND AREAS AND COMPARE
    head_box_area = cv2.contourArea(head_box_points)
    head_area = cv2.contourArea(approx_head)
    whole_box_area = cv2.contourArea(whole_box)
    try:
        head_box_ratio = head_area / head_box_area
        box_whole_ratio = head_box_area / whole_box_area
    except ZeroDivisionError:
        output[0] = "screw"
        return tuple(output)
    if head_box_ratio >= .9:
        output[1] = ("head_box_ratio", head_box_ratio)
        output[0] = "bolt"
        # return tuple(output)  # COMMENT OR UNCOMMENT?
    if box_whole_ratio >= .6:
        output[1] = ("whole box ratio", box_whole_ratio)
        output[0] = "other"
        return tuple(output)
        '''

    # RETURN VALUE BASED ON NUMBER OF VERTICAL LINES
    # """
    num_vertical, edges = count_parallel_lines(img, head_box, tolerance=0.2)
    output[4] = edges

    output[1] = ("vertical lines", num_vertical)
    if num_vertical >= 2:
        output[0] = "bolt"
        return(tuple(output))
    else:
        output[0] = "screw"
        return(tuple(output))
    # """


def mat_line(pt1, pt2):
    """
    matrix representation of a line based off of standard form formula:
    -(y2 - y1)*x + (x2 - x1)*y = x2*y1 - x1*y2
    """
    x1, y1 = pt1
    x2, y2 = pt2

    return np.array([-(y2 - y1), x2 - x1]), np.array([x2*y1 - x1*y2])


def contour_to_mat(contour):
    output = np.zeros([contour.shape[0], 2])
    for i, point in enumerate(contour):
        output[i][0] = point[0][0]
        output[i][1] = point[0][1]
    return output


def add_affine(vec, axis=0):
    """
    assumes 2D input
    """
    ones_shape = [None, None]
    ones_shape[axis] = vec.shape[axis] + 1
    ones_shape[1 - axis] = vec.shape[1 - axis]

    output = np.ones(ones_shape, dtype=int)

    output[: vec.shape[0], : vec.shape[1]] = vec

    return output


def straighten_image(image, thresh, contour):
    """
    straighten about least squares line and center, then crop
    if at the end the line of concavity is vertical, it's a nut
    """
    img = image.copy()
    thresh = thresh.copy()
    cnt = contour.copy()
    cnt = contour_to_mat(cnt)

    center, dims, angle = cv2.minAreaRect(contour)
    width = int(dims[0])
    height = int(dims[1])

    src_pts = cv2.boxPoints((center, dims, angle)).astype("float32")
    dst_pts = np.array([[0,       height-1],
                        [0,       0       ],
                        [width-1, 0       ],
                        [width-1, height-1]], dtype="float32")
    if width > height:
        dst_pts = np.flip(dst_pts, axis=1)

    M0 = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(image, M0, (min(width, height), max(width, height)))
    thresh = cv2.warpPerspective(thresh, M0, (min(width, height), max(width, height)))
    cnt = (M0 @ add_affine(cnt, axis=1).transpose())[:-1, :].transpose().astype(int)

    width = img.shape[1]
    height = img.shape[0]

    cnt_top, _ = cv2.findContours(thresh[0: height // 2, : ], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_bottom, _ = cv2.findContours(thresh[height // 2: , : ], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cv2.contourArea(cnt_bottom[0]) > cv2.contourArea(cnt_top[0]):
        M2 = cv2.getRotationMatrix2D((width // 2, height // 2), 180, 1)
        img = cv2.warpAffine(img, M2, (width, height))
        cnt = (M2 @ add_affine(cnt, axis=1).transpose()).transpose().astype(int)

    return img, cnt


def int_string_format(num, digits=3, padding=" "):
    num_str = str(num)
    to_add = digits - len(str(num))

    for _ in range(to_add):
        num_str = padding + num_str

    return num_str


def count_parallel_lines(image, rectangle, long_base=True, direc=np.array([1, 0]),
                         t_lower=600, t_upper=700, aperture_size=5,
                         blur=(1, 1), tolerance=0.1):
    """
    checks if an image (likely canny edge image) has lines parallet to a certain direction within a
    bounding rectangle, given that the rectangle is on it's base
    returns number of lines perpendicular to direc
    """
    direc = direc / np.linalg.norm(direc)

    width = int(rectangle[1][0])
    height = int(rectangle[1][1])
    src_pts = np.int0(cv2.boxPoints(rectangle)).astype("float32")
    # coordinate of the points in box points after the rectangle has been straightened
    dst_pts = np.array([[0,       height-1],
                        [0,       0       ],
                        [width-1, 0       ],
                        [width-1, height-1]], dtype="float32")
    if (long_base and height > width) or (not long_base and width > height):
        dst_pts = np.flip(dst_pts, axis=1)

    M1 = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # print(M1.shape)


    # directly warp the rotated rectangle to get the straightened rectangle
    img_rot = cv2.warpPerspective(image, M1, (max(width, height), min(width, height)))


    # CANNY EDGE DETECTION ON ROTATED HEAD
    edges = cv2.Canny(cv2.blur(img_rot, blur), t_lower, t_upper, apertureSize=aperture_size)


    # HOUGH LINE APPROXIMATION
    num_vertical = 0
    try:
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=5,
            minLineLength=.2*img_rot.shape[0],
            maxLineGap=0.15*img_rot.shape[0])

        line_list = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                line_list.append(mat_line((x1, y1), (x2, y2)))
                vector = np.array([x2 - x1, y2 - y1])
                # dot product of the unit vectors; direc is already a unit vector
                dpu = np.dot(vector, direc) / np.linalg.norm(vector)
                if abs(dpu) < tolerance:
                    cv2.line(img_rot, (x1,y1), (x2,y2), (255,0,0), 2)

                    num_vertical += 1

        return num_vertical, edges

    except:
        return 0, img_rot


if __name__ == "__main__":
    print('\033c')

    img = cv2.imread(r"./images/wood_screw.jpg")
    # fastener_type, ratio, sketches, thresholds, head = screw_bolt_other(img); print(fastener_type, ratio)
    fastener_type, ratio, sketches, thresholds, head = screw_bolt_other(img, light_threshold = 230, thresholding=cv2.THRESH_BINARY_INV); print(fastener_type, ratio)

    # HANDLE WINDOWS
    # """
    cv2.imshow("original", img)
    cv2.imshow('sketches', sketches)
    cv2.imshow('thresholds', thresholds)
    if head is not None:
        cv2.imshow('head', head)
    # """


    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
def least_squares_perp_offset(contour):
    """
    returns the slope of the PCA line in vector form (2D only)
    """
    points = contour_to_mat(contour)
    n = points.shape[1]
    splitted = np.hsplit(points, 2)
    x, y = splitted[0], splitted[1]
    y_bar = np.mean(y)
    x_bar = np.mean(x)
    try:
        B = 1/2 * (y.transpose() @ y - n * y_bar**2 - x.transpose() @ x + n * x_bar**2) / \
            (n * x_bar * y_bar - x.transpose() @ y)
        B = B[0].item()
    except ZeroDivisionError:
        return np.array([0, 1])
    return np.array([1, -B + np.sqrt(B*B + 1)])
def mat_to_contour(mat):
    output = np.zeros([mat.shape[0], 1, 2])
    for i, point in enumerate(mat):
        output[i][0][0] = point[0]
        output[i][0][1] = point[1]
    return output
def find_intersect(eq1, eq2):
    """
    each eq_ is a tuple outputted by mat_line
    """
    coeffs = np.vstack((eq1[0], eq2[0]))
    vals = np.vstack((eq1[1], eq2[1]))
    try:
        sol = np.linalg.inv(coeffs) @ vals
        return sol[0].item(), sol[1].item()
    except:
        return None
def intersection_in_range(inter_point, pt1, pt2):
    if inter_point is None:
        return False
    x = inter_point[0]
    y = inter_point[1]
    min_x, max_x = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
    min_y, max_y = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])
    return (min_x <= x and x <= max_x) and (min_y <= y <= max_y)
def intersection_in_range_2(eq1, eq2, pt1, pt2):
    return intersection_in_range(find_intersect(eq1, eq2), pt1, pt2)
def rotate_image(rotateImage, angle):
    """
    purely for testing purposes
    """
    # Taking image height and width
    imgHeight, imgWidth = rotateImage.shape[0], rotateImage.shape[1]
    # Computing the centre x,y coordinates
    # of an image
    centreY, centreX = imgHeight//2, imgWidth//2
    # Computing 2D rotation Matrix to rotate an image
    rotationMatrix = cv2.getRotationMatrix2D((centreY, centreX), angle, 1.0)
    # Now will take out sin and cos values from rotationMatrix
    # Also used numpy absolute function to make positive value
    cosofRotationMatrix = np.abs(rotationMatrix[0][0])
    sinofRotationMatrix = np.abs(rotationMatrix[0][1])
    # Now will compute new height & width of
    # an image so that we can use it in
    # warpAffine function to prevent cropping of image sides
    newImageHeight = int((imgHeight * sinofRotationMatrix) +
                         (imgWidth * cosofRotationMatrix))
    newImageWidth = int((imgHeight * cosofRotationMatrix) +
                        (imgWidth * sinofRotationMatrix))
    # After computing the new height & width of an image
    # we also need to update the values of rotation matrix
    rotationMatrix[0][2] += (newImageWidth/2) - centreX
    rotationMatrix[1][2] += (newImageHeight/2) - centreY
    # Now, we will perform actual image rotation
    rotatingimage = cv2.warpAffine(
        rotateImage, rotationMatrix, (newImageWidth, newImageHeight))
    return rotatingimage
'''
