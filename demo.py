import os, cv2
from utils import coords
from utils import ocr
from utils import img_processing
from picamera import PiCamera


def main():
    camera = PiCamera()
    video = cv2.VideoCapture(camera)

    while(True):
        ret, frame = video.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv2.imshow('frame', gray)


        hand_coord = coords.get_hand_coord(frame)[0]
        words_coord = coords.get_word_coord(frame)
        touchable_words_coord = coords.get_touchable_word_coord(
            hand_coord, words_coord)
        img = cv2.imread(frame)
        img_processing.draw_box_on_image_green(img, tuple(hand_coord[0]),
                                                tuple(hand_coord[1]))

        img_processing.draw_box_on_image_red(img, touchable_words_coord[0],
                                                touchable_words_coord[1])
        # cv2.imwrite(RESULT_IMG_PATH, img) result part
        roi = img_processing.cut_image(img, touchable_words_coord[0],
                                        touchable_words_coord[1])
        text = ocr.naver(roi)
        # # f = open(RESULT_TXT_PATH, 'w')
        # f.write(text)
        # f.close()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()