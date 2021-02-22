import cv2

print("start")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

width = 1280
height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('your_video.avi', fourcc, 20.0, size)
print("init stop")

try:
    while(True):
        print("frame")
        _, frame = cap.read()
        cv2.imshow('Recording...', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Video stoped and saved")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
