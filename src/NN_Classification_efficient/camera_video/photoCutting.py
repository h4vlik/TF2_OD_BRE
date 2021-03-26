import cv2
import os


def centerCrop(image, w, h):
    # input image size
    width = 1280
    height = 720    # Get dimensions

    new_width = w
    new_height = h

    left = int((width - new_width)//2)
    top = int((height - new_height)//2)
    right = int((width + new_width)//2)
    bottom = int((height + new_height)//2)

    # Crop the center of the image
    # image = image.crop((left, top, right, bottom))
    image = image[top:bottom, left:right]

    return image


def optionalCrop(image, w, h):
    # ____ dimensions ______
    # croped region
    y = 100
    x = 250

    top = y
    bottom = y + h
    left = x
    right = x + w

    crop = frame[top:bottom, left:right]

    return crop


# define path to DIPLOMKA FILE
main_dir_path = os.path.dirname(os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)))))))


# path to video read
path_to_video = os.path.join(
    main_dir_path,
    r'DATA\\Camera_data\\MERENI_12_3_LESNA_VIDEO',
    os.path.basename("lesna_01.mp4"))


# path to save images
path_to_save_file = os.path.join(
    main_dir_path,
    r'DATA\\DISPLAY_IMAGES\\FOR_FUSION\\IBC_15_3\\ibc_01\\')

path_to_video_save = os.path.join(
    main_dir_path,
    r'KODING\\TF2_OD_BRE\\data\\Camera_input_data\\video\\',
    os.path.basename("lesna_01_upravene.mp4"))


# ____ dimensions ______
# croped region
h = 500
w = 500

size = (w, h)
# ______________________

# video read capture
cap = cv2.VideoCapture(path_to_video)

# set video save format
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(path_to_video_save, fourcc, 15.0, size)

i = 0
while(True):
    ret, frame = cap.read()
    if ret is False:
        print("false")
        break

    #crop = centerCrop(frame, w, h)
    crop = optionalCrop(frame, w, h)

    out.write(crop)
    #cv2.imwrite(os.path.join(path_to_save_file, 'display'+str(i)+'.jpg'), crop)
    i += 1

cap.release()
out.release()
cv2.destroyAllWindows()
