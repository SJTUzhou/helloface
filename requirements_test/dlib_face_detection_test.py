import sys
import dlib
import cv2

def main():
    detector = dlib.get_frontal_face_detector() # get the face classifier

    # input the argument of the command line, input the image files the following arguments
    for f in sys.argv[1:]:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        # change BGR color image to Gray image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # use a detector to detect the faces, dets is the result
        dets = detector(gray_img, 1) 
        # get the number of faces
        print("Number of faces detected: {}".format(len(dets))) 
        # enumerate is a method to give the indexes to an interable object
        # face is a dlib.rectangle recording the position of face
        # left()、top()、right()、bottom() return the positions of 4 edges of an dlib.rectangle
        for index, face in enumerate(dets):
            print('face {}; left {}; top {}; right {}; bottom {}'.format(index, face.left(), face.top(), face.right(), face.bottom()))

            # show the faces in the image
            left = face.left()
            top = face.top()
            right = face.right()
            bottom = face.bottom()
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.imshow("Dlib face detection test", img)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()