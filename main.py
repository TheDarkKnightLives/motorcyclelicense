import cv2
import time
from my_functions import *

# Set source as webcam
source = 0

save_video = True  # Set to True if you want to save video
show_video = True  # Set to True to display video
save_img = False  # Set to True if you want to save image frames

# Setup video writer if saving the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
frame_size = (640, 480)  # Define frame size (modify as needed)
out = cv2.VideoWriter('output.avi', fourcc, 20.0, frame_size)

# Open webcam
cap = cv2.VideoCapture(source)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.resize(frame, frame_size)  # Resize frame if necessary
        orifinal_frame = frame.copy()
        frame, results = object_detection(frame)

        rider_list = []
        head_list = []
        number_list = []

        # Sort results into rider, head, and number lists
        for result in results:
            x1, y1, x2, y2, cnf, clas = result
            if clas == 0:
                rider_list.append(result)
            elif clas == 1:
                head_list.append(result)
            elif clas == 2:
                number_list.append(result)

        for rdr in rider_list:
            time_stamp = str(time.time())
            x1r, y1r, x2r, y2r, cnfr, clasr = rdr
            for hd in head_list:
                x1h, y1h, x2h, y2h, cnfh, clash = hd
                if inside_box([x1r, y1r, x2r, y2r], [x1h, y1h, x2h, y2h]):  # Check if the head is inside the rider's bbox
                    try:
                        head_img = orifinal_frame[y1h:y2h, x1h:x2h]
                        helmet_present = img_classify(head_img)  # Classify whether helmet is present
                    except:
                        helmet_present = [None]  # In case of failure, default to None

                    if helmet_present[0] == True:  # Helmet detected
                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 0), 1)
                        frame = cv2.putText(frame, 'Helmet', (x1h, y1h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    elif helmet_present[0] == None:  # Uncertain result
                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 255, 255), 1)
                        frame = cv2.putText(frame, 'Unknown', (x1h, y1h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    elif helmet_present[0] == False:  # No helmet detected
                        frame = cv2.rectangle(frame, (x1h, y1h), (x2h, y2h), (0, 0, 255), 1)
                        frame = cv2.putText(frame, 'No Helmet', (x1h, y1h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        try:
                            # Save image of the rider
                            cv2.imwrite(f'riders_pictures/{time_stamp}.jpg', frame[y1r:y2r, x1r:x2r])
                        except:
                            print('Could not save rider')

                        for num in number_list:
                            x1_num, y1_num, x2_num, y2_num, conf_num, clas_num = num
                            if inside_box([x1r, y1r, x2r, y2r], [x1_num, y1_num, x2_num, y2_num]):
                                try:
                                    # Save image of number plate
                                    num_img = orifinal_frame[y1_num:y2_num, x1_num:x2_num]
                                    cv2.imwrite(f'number_plates/{time_stamp}_{conf_num}.jpg', num_img)
                                except:
                                    print('Could not save number plate')

        if save_video:  # Save video frame
            out.write(frame)
        if save_img:  # Save image frame
            cv2.imwrite('saved_frame.jpg', frame)
        if show_video:  # Display video frame
            frame = cv2.resize(frame, (900, 450))  # Resize to fit screen
            cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on pressing 'q'
            break
    else:
        break

cap.release()  # Release webcam
out.release()  # Release video writer if saving video
cv2.destroyAllWindows()  # Close all OpenCV windows
print('Execution completed')
