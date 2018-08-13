import os
import time
import re
import csv
import cv2
import numpy as np
import time
from collections import defaultdict
from imutils import face_utils
from face_roi import FaceROI # used to create a face roi from PAF facial features
from tiny_face.tiny_face_detect import TinyFace  # used to find a tight face bounding box from the face ROI
from headpose.head_pose import HopeHeadPose  # used to find the head pose
from annotation_reader import Annotation
from frame_reader import FrameReader

class Train:

    def __init__(self, output_training_data):
        '''
        Constructor removes previous file name if it exists and writes the headers
        of the output features file.

        '''
        self.output_training_data = output_training_data

        try:
            os.remove(self.output_training_data)
        except OSError:
            pass

        # write to csv (by appending to file)
        with open(self.output_training_data, 'a') as outfile:
            csv_writer = csv.writer(outfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['subdir','frame_num','face_startX','face_startY','face_endX','face_endY',\
            'face_height','face_width','face_x_center','face_y_center','yaw','pitch','roll','attention_label'])

    def group_data_by_frame(self, annotation_rows, frames_set):
        row_data_by_frame = defaultdict(list)

        # group annotations by frame number (1 frame can have many faces in a frame)
        # iterate through each annotation row, put into a dictionary, with keys by
        # frame name, values as a list of annotations for the frame
        for annotation_row_data in annotation_rows:
            subdir = annotation_row_data[0]
            frame_num = str(int(annotation_row_data[1]))
            annotation_row_name = subdir + '/' + frame_num

            if annotation_row_name in frames_set:
                row_data_by_frame[annotation_row_name].append(annotation_row_data)

        return row_data_by_frame

    def get_file_path(self, frames_dir_path, subdir, frame_num):

        frame_num_len = len(str(frame_num))
        formatted_frame_num = ''
        frame_num_len = len(str(frame_num))

        # need to append .jpg to frame num, which has either 4 or 5 digits
        if frame_num_len < 5:  # less than 10k frames in subdir
            formatted_frame_num = formatted_frame_num = str(frame_num).zfill(4) + '.jpg'
        elif frame_num_len > 4:  # might be 10k+ frames in a subdir
            formatted_frame_num = str(frame_num) + '.jpg'

        # create path with full format to find frame
        file_path = '/'.join([frames_dir_path, subdir, formatted_frame_num])
        return file_path

    def get_head_pose(self, frame, face_bounding_box, headposer, display_frame):

        # if face found
        if face_bounding_box:
            startX = face_bounding_box[0]
            startY = face_bounding_box[1]
            endX = face_bounding_box[2]
            endY = face_bounding_box[3]
            bbox_width = abs(endX - startX)
            bbox_height = abs(endY - startY)
            yaw, pitch, roll = headposer.get_pose(frame, face_bounding_box, display_frame) # get pose
            head_pose = (yaw, pitch, roll)

            if display_frame:
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                out_text = 'yaw: '+ str(yaw) + ' pitch: ' + str(pitch) + ' roll: ' + str(roll)
                cv2.putText(frame, out_text, (startX, startY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        else:  # no bounding box found, but still write something
            if display_frame:
                out_text = 'No face found'
                cv2.putText(frame, out_text, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)

        return head_pose

    def get_face_from_roi(self, frame, face_ROI, tiny_face_detector, display_frame):
        # get face bb with CNN
        face_bounding_box = tiny_face_detector.get_face_box(frame, face_ROI)
        # if display, show the original ROI box
        if display_frame:
            roi_x1, roi_y1, roi_x2, roi_y2 = face_ROI
            # draw the original face ROI bounding box in blue
            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2),
                (255, 0, 0), 2)
        return face_bounding_box

    def build_training_data(self, frames_dir_path, annotations_path, frame_size, display_frame):

        '''
        This function will find matches between the frame directory path and annotations,
        and will run feature extraction for those matches only. Feature extraction includes
        face box estimation and head pose. Finally, it will write the features to csv.

        '''
        print('building training data...')

        frame_reader = FrameReader()  # reads the frames from a directory
        annotator = Annotation(annotations_path, frame_size)  # reads annotations from csv
        face_roi = FaceROI()  # creates face roi from face features (from annotations)
        tiny_face_detector = TinyFace()  # creates tight face box from the face roi
        headposer = HopeHeadPose()  # object gets the headpose (from the tight bounding box)

        # retrieve a set with all the relative frame paths in the directories to get features for testing/training
        frames_set = frame_reader.create_frames_set(frames_dir_path)
        annotation_rows = annotator.get_annotation_rows(annotations_path) # a list of all the annotations

        # group annotations by frame in a dict
        row_data_by_frame = self.group_data_by_frame(annotation_rows, frames_set)

        # iterate through frames in the annotations dict
        for frame_name, list_of_data in row_data_by_frame.items():

            subdir = frame_name.split('/')[0]
            frame_num = frame_name.split('/')[1]
            file_path = self.get_file_path(frames_dir_path, subdir, frame_num)

            # iterate through each annotated face in the frame
            for row_data in list_of_data:
                # look up facial keypoints and attn labels
                face_keypoints = row_data[6:18]  # see annotations file for legend
                raw_bounding_box = row_data[2:6]  # normalized, need to convert to frame size
                attention_label = row_data[18]
                start_time = time.time()  # start the timer
                # load the frame
                frame = frame_reader.load_frame(file_path)
                frame = cv2.resize(frame, (frame_size[1], frame_size[0]))  # resize to user dimensions

                head_pose = None  # initialize as empty
                face_bounding_box = None
                # get face ROI to reduce size for Tiny Face detector to search with CNN
                face_ROI = face_roi.create_roi_from_features(face_keypoints, frame, frame_size)

                # if got a face ROI
                if face_ROI:
                    face_bounding_box = self.get_face_from_roi(frame, face_ROI, tiny_face_detector, display_frame)

                # if face bounding box found, then get the head pose
                if face_bounding_box:
                    head_pose = self.get_head_pose(frame, face_bounding_box, headposer, display_frame)

                # predict label here with hard coded thresholds

                if display_frame:
                    # show the output image
                    smaller = cv2.resize(frame, None, fx=0.90, fy=0.90)
                    winname = "{}".format(file_path)
                    cv2.namedWindow(winname)        # Create a named window
                    cv2.moveWindow(winname, 40,30)  # Move it to (40,30)
                    cv2.imshow("{}".format(winname), smaller)
                    key = cv2.waitKey(0) & 0xFF
                    cv2.destroyAllWindows()

                # save all features and attention label to csv here
                self.write_to_csv(subdir,frame_num,face_bounding_box,head_pose,attention_label)
                print('process time for face: {:.2f} secs'.format(time.time() - start_time))

    def write_to_csv(self,subdir,frame_num,face_bounding_box,head_pose,attention_label):

        '''
        writes the relevant features to csv.

        '''
        frame_num_len = len(str(frame_num))
        formatted_frame_num = ''
        frame_num_len = len(str(frame_num))
        frame_num = str(int(frame_num))  # need to just isolate the number, remove 0s and make str

        # default values for face in case face not found
        face_height = 'NA'
        face_width = 'NA'
        face_x_center = 'NA'
        face_y_center = 'NA'
        yaw = 'NA'
        pitch = 'NA'
        roll = 'NA'
        face_startX = 'NA'
        face_startY = 'NA'
        face_endX = 'NA'
        face_endY = 'NA'

        # if face found, then get the width, height, and centroid of face box
        if face_bounding_box:
            face_startX, face_startY, face_endX, face_endY = face_bounding_box
            face_height = face_endY - face_startY
            face_width = face_endX - face_startX
            face_x_center = int(face_width/2 + face_startX)
            face_y_center = int(face_height/2 + face_startY)
        if head_pose:
            yaw = head_pose[0]
            pitch = head_pose[1]
            roll = head_pose[2]
        # write to csv (by appending to file)
        with open(self.output_training_data, 'a') as outfile:
            csv_writer = csv.writer(outfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow([subdir,frame_num,face_startX,face_startY,face_endX,face_endY,face_height,\
            face_width,face_x_center,face_y_center,yaw,pitch,roll,attention_label])

def main(test_frames_path, annotations_path, frame_size, output_file_name, display_frame):

    start = time.time()

    trainer = Train(output_file_name)
    trainer.build_training_data(test_frames_path, annotations_path, frame_size, display_frame)

    print("time {:.2f} mins total for training".format( (time.time()-start)/60.0  ) )

if __name__ == "__main__":
    '''

    Will write to csv face bounding boxes given a directory of test frames (with subdirectory of households),
    and an annotation file that has the body bounding boxes.

    Note, the directory of frames need to fit the format:

        test-frames -> household_number_and_hour -> frame_number.jpg

    '''
    display_frame = True
    test_frames_path = 'missing-frames'
    annotations_path = 'Inderbir-annotations.csv'
    output_file_name = 'output_training_data.csv'
    frame_size = (1080,1920)  # set to what the actual image frame size is (note, optimized for 1080)
    # frame_size = (720,1280)  # less accurate face detection at this resolution

    main(test_frames_path, annotations_path, frame_size, output_file_name, display_frame)
