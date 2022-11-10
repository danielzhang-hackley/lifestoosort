import cv2
import numpy as np


def screw_bolt_other(image):
    output = [None for _ in range(5)]  # type, ratio, drawings, thresh, head

    # PROCESS IMAGE AND CREATE CONTOURS
    img = image.copy()  # keep the original image clean
    grayscale = cv2.cvtColor(cv2.blur(img, (5, 5)), cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(grayscale, 220, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output[2], output[3] = img, thresh


    # CREATE CONTOURS
    if len(contours) > 0:
        cnt = contours[0]
        max_area = cv2.contourArea(cnt)
        for cont in contours:
            if cv2.contourArea(cont) > max_area:
                cnt = cont
                max_area = cv2.contourArea(cont)

        whole_box = np.int0(cv2.boxPoints(cv2.minAreaRect(cnt)))
        
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
        else:
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
            approx = np.vstack([approx[: min(notch1_idx + 1, notch2_idx + 1)],
                                approx[max(notch1_idx, notch2_idx):]])
            rect = cv2.minAreaRect(approx)
            head_box = cv2.boxPoints(rect)
            head_box = np.int0(head_box)


            # DRAW EVERYTHING ELSE
            img = cv2.circle(img, notch1, radius=5, color=(0, 255, 0), thickness=-1)
            img = cv2.circle(img, notch2, radius=5, color=(0, 255, 0), thickness=-1)
            cv2.drawContours(img,[head_box], 0, (255, 0, 255), 2)
            cv2.drawContours(img,[whole_box], 0, (255, 255, 0), 2)

            # FIND AREAS AND COMPARE
            head_box_area = cv2.contourArea(head_box)
            head_area = cv2.contourArea(approx)
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

            # ROTATE IMAGE ABOUT CENTER OF BOUNDING BOX
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = head_box.astype("float32")
            # coordinate of the points in box points after the rectangle has been straightened
            dst_pts = np.array([[0,       height-1],
                                [0,       0       ],
                                [width-1, 0       ],
                                [width-1, height-1]], dtype="float32")
            if height > width:
                dst_pts = np.flip(dst_pts, axis=1)

            # the perspective transformation matrix
            M1 = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            img_rot = cv2.warpPerspective(image, M1, (max(width, height), min(width, height)))
            # cv2.imshow("head", img_rot)


            # CANNY EDGE DETECTION ON ROTATED HEAD
            t_lower = 600  # Lower Threshold
            t_upper = 700  # Upper threshold
            aperture_size = 5  # Aperture size; TODO: adjust these three values
            edges = cv2.Canny(cv2.blur(img_rot, (1,1)), t_lower, t_upper, apertureSize=aperture_size)
            output[4] = edges


            # HOUGH LINE APPROXIMATION
            num_vertical = 0
            try:
                lines = cv2.HoughLinesP(
                    edges,
                    rho=1,
                    theta=np.pi/180,
                    threshold=5,
                    minLineLength=.33*img_rot.shape[:2][0],
                    maxLineGap=0.2*img_rot.shape[:2][0])

                for line in lines:
                    for x1, y1, x2, y2 in line:
                        vector = np.array([x2 - x1, y2 - y1])
                        # dot product of the unit vectors
                        dpu = np.dot(vector, np.array([1, 0])) / (np.linalg.norm(vector) * np.linalg.norm(np.array([1, 0])))
                        # cv2.line(edges, (x1,y1), (x2,y2), (255,0,0), 3)
                        if dpu > -0.1 and dpu < 0.1:
                            num_vertical += 1
            except:
                output[0], output[1] = "screw", 0
                return tuple(output)

            # cv2.imshow("head", img_rot)

            # RETURN VALUE BASED ON NUMBER OF VERTICAL LINES
            output[1] = ("vertical lines", num_vertical)
            if num_vertical >= 2:
                output[0] = "bolt"
                return(tuple(output))
            else:
                output[0] = "screw"
                return(tuple(output))


if __name__ == "__main__":
    print('\033c')

    img = cv2.imread(r'./images/hex_bolt.jpg')
    fastener_type, ratio, sketches, thresholds, head = screw_bolt_other(img)

    print(fastener_type, ratio)

    # HANDLE WINDOWS
    cv2.imshow("original", img)
    cv2.imshow('sketches', sketches)
    cv2.imshow('thresholds', thresholds)
    if head is not None:
        cv2.imshow('head', head)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
