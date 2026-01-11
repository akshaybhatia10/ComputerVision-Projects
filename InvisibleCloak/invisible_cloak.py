import cv2
import numpy as np
import time

def main():
    print("Welcome to the Invisible Cloak!")
    print("Instructions:")
    print("1. The camera will capture the background for 3 seconds.")
    print("   Please move out of the frame!")
    print("2. Once started, anything RED will become invisible.")
    print("3. Press 'q' to quit.")

    cap = cv2.VideoCapture(0)
    time.sleep(3) # Give time to user to move aside
    background = 0

    # Capturing the background
    for i in range(60):
        ret, background = cap.read()
    
    background = np.flip(background, axis=1)

    while(cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
            
        img = np.flip(img, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Generating mask to detect red color
        # HSV values for Red can wrap around, so we need two ranges.
        
        # Range 1 (0-10)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        # Range 2 (170-180)
        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask1 = mask1 + mask2 # OR operation

        # Refining the mask
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Creating an inverted mask to segment out the red color
        mask2 = cv2.bitwise_not(mask1)

        # Segmenting the red color part out of the frame using bitwise and with the inverted mask
        res1 = cv2.bitwise_and(img, img, mask=mask2)

        # Creating image showing static background frame pixels only for the masked region
        res2 = cv2.bitwise_and(background, background, mask=mask1)

        # Generating the final output
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

        cv2.imshow("Invisible Cloak", final_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
