import numpy as np
import imutils
import cv2

PATH = "video.mp4"
firstFrame = None
min_area = 500


def detector(frame):
    boxes = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    global firstFrame
    if firstFrame is None:
        firstFrame = gray

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        [x, y, w, h] = cv2.boundingRect(c)
        #         print([x, y, w, h])
        boxes.append([x, y, w, h])

    return boxes


if __name__ == '__main__':

    cap = cv2.VideoCapture(PATH)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f'fps: {fps}')
    timer = 0
    timer_position = (20, 50)

    totFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'total frames: {totFrames}')

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        #     print(f'frame: {timer}')

        box_locations = detector(frame)

        tempImg = frame.copy()
        maskShape = (frame.shape[0], frame.shape[1], 1)
        mask = np.full(maskShape, 0, dtype=np.uint8)

        for x, y, w, h in box_locations:
            #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            left, top, right, bottom = x, y, x + w, y + h

            tempImg[top:bottom, left:right] = cv2.blur(tempImg[top:bottom, left:right], (50, 50))

            start_point = (left, top)
            end_point = (right, bottom)

            cv2.rectangle(tempImg,
                          start_point,
                          end_point,
                          color=(255),
                          thickness=2)

            cv2.rectangle(mask,
                          start_point,
                          end_point,
                          color=(255),
                          thickness=-1)

        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        img2_fg = cv2.bitwise_and(tempImg, tempImg, mask=mask)
        dst = cv2.add(img1_bg, img2_fg)

        #     dst = frame

        cv2.putText(
            dst,
            str(int(timer / fps)),
            timer_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (209, 80, 0, 255),
            3)

        cv2.imshow('Video', dst)

        timer += 1

        # TODO: SAVE TO LIST TO IGNORE RAM AND CACHE ISSUES AND THEN PLAY IN ORIGINAL FRAME RATE
        if cv2.waitKey(25) & 0xFF == ord('q'):  # int(1000 / fps)
            break

    cap.release()
    cv2.destroyAllWindows()
