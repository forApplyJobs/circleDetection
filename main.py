import cv2
import numpy as np

cap = cv2.VideoCapture("deneme8.avi")


def colored_histogram_equalizer(img):
    new_img = img.copy()
    img_yuv = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def image_process1(cap, count, isFound, center, area):  # bura Ece nin kod
    # İlk kamera işlemleri.
    area_i = 0
    ret, frame_ = cap.read()
    cv2.waitKey(1)
    threshold = 25
    obj_color1 = 110
    green_color1 = 60
    frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    shape = frame.shape  # shape[0] = y coord , shape[1] = x coord
    range_center = shape[1] / 2, shape[0] / 2
    # Araç kamerası için Merkez noktası belirlenir range1=[x_center,y_center]
    if isFound == True:
        mask = np.zeros(shape, dtype="uint8")
        cv2.circle(mask, (center[0], center[1]), area + threshold, 255, -1)
        frame_ = cv2.bitwise_and(frame_, frame_, mask=mask)
        cv2.imshow("masked", frame_)
        frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
        obj_color1 += 50
        green_color1 += 20

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Color filtering parameters
    # Use blobColor = 0 to extract dark blobs and blobColor = 255 to extract light blobs
    params.filterByColor = False
    params.blobColor = 0

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 3000
    params.maxArea = (frame.shape[0] * frame.shape[1]) / 2

    # Set threshold filtering parameters
    # params.minThreshold = 0
    # params.maxThreshold = 20000000

    # Set Circularity filtering parameters min 0, max 1
    params.filterByCircularity = True
    params.minCircularity = 0.55
    params.maxCircularity = 1

    # Set Convexity filtering parameters min 0, max 1
    params.filterByConvexity = True
    params.minConvexity = 0.3
    params.maxConvexity = 1

    # Set inertia filtering parameters min 0, max 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(frame)

    # get coordinates and sizes of detected objects
    coordinates = [key_point.pt for key_point in keypoints]
    sizes = [key_point.size for key_point in keypoints]

    # check colors, eliminate blue ones
    eliminate_idx = []
    obj_color = 0
    green_color = 0
    for j in range(0, len(keypoints)):
        obj_color, green_color, a_ = frame_[int(coordinates[j][1]), int(
            coordinates[j][0])]  # image color code is BGR in my case, change channel if it is necessary
        # print("B:"+str(obj_color)+"G:"+str(green_color)+"R:"+str(a_))
        if obj_color > obj_color1 or green_color > green_color1:
            eliminate_idx.append(j)

    # remove some elements of a list
    keypoints = [i for j, i in enumerate(keypoints) if j not in eliminate_idx]
    coordinates = [i for j, i in enumerate(coordinates) if j not in eliminate_idx]
    sizes = [i for j, i in enumerate(sizes) if j not in eliminate_idx]
    if coordinates:
        count += 1
        print(count)
        center2 = [int(coordinates[0][0]), int(coordinates[0][1])]
        text = "X :" + str(int(coordinates[0][0])) + " " + "Y :" + str(int(coordinates[0][1])) + " " + "R :" + str(
            sizes[0])
        cv2.putText(frame_, text, (0, 150), 1, 1, (0, 255, 0), 1)
        cv2.circle(frame_, (int(coordinates[0][0]), int(coordinates[0][1])), int(sizes[0] / 2), (255, 255, 0), 3)
        area_i = int(sizes[0])
    else:
        area_i = 0
        center2 = [None, None]
    cv2.imshow("frame", frame_)
    return count, center2, area_i
    # Draw blobs on our image as red circles
    # cv2.rectangle(frame_, (int((shape[1]/2)-80), int((shape[0]/2)-80)), (int((shape[1]/2) +80),int((shape[0]/2) + 20)), (0, 255, 0), 2) # son değerler sırayla sabitlenmek istenen basınç değeri,Area


def image_process2(cap, count, isFound, center2, area):  # bura Ece nin kod
    center = [None, None]
    ret, frame = cap.read()

    # shape[0] = y coord , shape[1] = x coord
    # Araç kamerası için Merkez noktası belirlenir range1=[x_center,y_center]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    shape = hsv.shape
    a_total = 0
    b_total = 0
    r_total = 0
    threshold = 25
    if isFound == True:
        """mask = np.zeros(shape[:2], dtype='uint8')
        cv2.circle(mask, (center2[0], center2[1]), area+threshold, 255, -1)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame=cv2.equalizeHist(frame)
        detected_circles = cv2.HoughCircles(frame,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=50, minRadius=100, maxRadius=300)"""
        blank = np.zeros(hsv.shape[:2], dtype='uint8')
        removed = lineDetected(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # removed=cv2.Canny(gray, 10, 25, apertureSize=3)
        contours, hierarchy = cv2.findContours(removed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > 10 and cv2.contourArea(contours[i]) < 1000:
                cp = contours[i]
                cv2.drawContours(blank, [cp], 0, (255, 255, 255), 3)
        detected_circles = cv2.HoughCircles(blank,
                                            cv2.HOUGH_GRADIENT, 1, 20, param1=20,
                                            param2=15, minRadius=50, maxRadius=500)
    else:
        blank = np.zeros(hsv.shape[:2], dtype='uint8')
        removed = lineDetected(frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # removed=cv2.Canny(gray, 10, 25, apertureSize=3)
        contours, hierarchy = cv2.findContours(removed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > 20 and cv2.contourArea(contours[i]) < 1000:
                cp = contours[i]
                cv2.drawContours(blank, [cp], 0, (255, 255, 255), 3)
        detected_circles = cv2.HoughCircles(blank,
                                            cv2.HOUGH_GRADIENT, 1, 20, param1=30,
                                            param2=15, minRadius=50, maxRadius=500)

    if detected_circles is not None:
        count += 1
        print(count)
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            a_total += a
            b_total += b
            r_total += r

        a_total = int(a_total / len(detected_circles[0, :]))
        b_total = int(b_total / len(detected_circles[0, :]))
        r_total = int(r_total / len(detected_circles[0, :]))
        # Draw the circumference of the circle.
        cv2.circle(frame, (a_total, b_total), r_total, (0, 255, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(frame, (a_total, b_total), 1, (0, 0, 255), 3)
        text = "X :" + str(a_total) + " " + "Y :" + str(b_total) + " " + "R :" + str(r_total)
        cv2.putText(frame, text, (0, 150), 1, 1, (0, 255, 0), 1)
        center = [a_total, b_total]
        area = r_total
        print("deneme")

    cv2.rectangle(frame, (int((shape[1] / 2) - 50), int((shape[0] / 2) - 50)),
                  (int((shape[1] / 2) + 70), int((shape[0] / 2) + 30)), (0, 255, 0), 2)
    cv2.waitKey(1)
    cv2.imshow("frame", frame)
    return count, center, area


def image_process3(cap, count, isFound, center, area):  # bura Ece nin kod
    # İlk kamera işlemleri.
    area_i = 0
    ret, frame_ = cap.read()
    cv2.waitKey(1)
    threshold = 25
    obj_color1 = 110
    green_color1 = 60
    frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
    shape = frame.shape  # shape[0] = y coord , shape[1] = x coord
    range_center = shape[1] / 2, shape[0] / 2
    # Araç kamerası için Merkez noktası belirlenir range1=[x_center,y_center]
    if isFound == True:
        mask = np.zeros(shape, dtype="uint8")
        cv2.circle(mask, (center[0], center[1]), area + threshold, 255, -1)
        frame_ = cv2.bitwise_and(frame_, frame_, mask=mask)
        cv2.imshow("masked", frame_)
        frame = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)
        obj_color1 += 50
        green_color1 += 20

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Color filtering parameters
    # Use blobColor = 0 to extract dark blobs and blobColor = 255 to extract light blobs
    params.filterByColor = False
    params.blobColor = 0

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 3000
    params.maxArea = (frame.shape[0] * frame.shape[1]) / 2

    # Set threshold filtering parameters
    # params.minThreshold = 0
    # params.maxThreshold = 20000000

    # Set Circularity filtering parameters min 0, max 1
    params.filterByCircularity = True
    params.minCircularity = 0.20
    params.maxCircularity = 1

    # Set Convexity filtering parameters min 0, max 1
    params.filterByConvexity = True
    params.minConvexity = 0.1
    params.maxConvexity = 1

    # Set inertia filtering parameters min 0, max 1
    params.filterByInertia = True
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(frame)

    # get coordinates and sizes of detected objects
    coordinates = [key_point.pt for key_point in keypoints]
    sizes = [key_point.size for key_point in keypoints]

    # check colors, eliminate blue ones
    eliminate_idx = []
    obj_color = 0
    green_color = 0
    for j in range(0, len(keypoints)):
        obj_color, green_color, a_ = frame_[int(coordinates[j][1]), int(
            coordinates[j][0])]  # image color code is BGR in my case, change channel if it is necessary
        # print("B:"+str(obj_color)+"G:"+str(green_color)+"R:"+str(a_))
        if obj_color > 0 or green_color > 0:
            eliminate_idx.append(j)

    # remove some elements of a list
    keypoints = [i for j, i in enumerate(keypoints) if j not in eliminate_idx]
    coordinates = [i for j, i in enumerate(coordinates) if j not in eliminate_idx]
    sizes = [i for j, i in enumerate(sizes) if j not in eliminate_idx]
    if coordinates:
        count += 1
        print(count)
        center2 = [int(coordinates[0][0]), int(coordinates[0][1])]
        text = "X :" + str(int(coordinates[0][0])) + " " + "Y :" + str(int(coordinates[0][1])) + " " + "R :" + str(
            sizes[0])
        cv2.putText(frame_, text, (0, 150), 1, 1, (0, 255, 0), 1)
        cv2.circle(frame_, (int(coordinates[0][0]), int(coordinates[0][1])), int(sizes[0] / 2), (255, 255, 0), 3)
        area_i = int(sizes[0])
    else:
        area_i = 0
        center2 = [None, None]
    cv2.imshow("frame", frame_)
    return count, center2, area_i
    # Draw blobs on our image as red circles
    # cv2.rectangle(frame_, (int((shape[1]/2)-80), int((shape[0]/2)-80)), (int((shape[1]/2) +80),int((shape[0]/2) + 20)), (0, 255, 0), 2) # son değerler sırayla sabitlenmek istenen basınç değeri,Area


def remove_lines(image_):
    image = image_.copy()
    kernel = np.ones((5, 5), np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Use canny edge detection
    edges = cv2.Canny(gray, 10, 25, apertureSize=3)

    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=50,  # Min number of votes for valid line
        minLineLength=50,  # Min allowed length of line
        maxLineGap=10  # Max allowed gap between line for joining them
    )
    if lines is not None:
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines joing the points
            # On the original image
            cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 0), 10)

            # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])
        return edges
    # Iterate over points

    else:
        return None


def lineDetected(image):
    # Read the input image
    cv2.waitKey(1)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ##########burası silinebilir filtre denemek amaçlı eklendi

    # Apply Gaussian blur to reduce high frequency noise
    gray = cv2.GaussianBlur(image, (3, 3), 0)
    # Use the Canny edge detector to find edges in the image
    edges = cv2.Canny(gray, 50, 150)
    lines = []
    # Run the Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=0, minLineLength=50, maxLineGap=10)
    # print(len[(lines)])

    # Iterate over the output lines and draw them on the image
    counter = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            counter += 1
            # Find the slope and intercept of the line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Compute the x-coordinates for each y-coordinate in the range of the line
            x = np.linspace(x1, x2, abs(y2 - y1) + 1).astype(int)
            y = (slope * x + intercept).astype(int)

            cv2.line(edges, (x1, y1), (x2, y2), -1, thickness=4)

    # print(counter)
    return edges


count = 0
count2 = 0
area = None
area2 = None
isFound = False
isFound2 = False
center = [None, None]
center2 = [None, None]
while True:
    # count,center,area=image_process1(cap,count,isFound,center,area)
    count2, center2, area2 = image_process1(cap, count2, isFound2, center2, area2)
    # if(center[0]!=None):
    # isFound=True
    # else:
    # isFound=False

    if (center2[0] != None):
        isFound2 = True
    else:
        isFound2 = False
