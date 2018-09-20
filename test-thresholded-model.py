# USAGE
# python test-thresholded-model.py --frame_directory test-frames --output_name_path predicted_labels_thresholded_rev2.csv --csv_results output_features_aug7.csv --one_to_two False --write_csv True

from sklearn.svm import SVC
import sklearn
import numpy as np
import pandas as pd
import math
import csv
from sklearn.metrics import classification_report, confusion_matrix
import os
from thresholded_model import Attention
import argparse
import cv2

class Hardcode:

    def __init__(self, output_name, write_csv):
        # if existing predictor file exists, remove it
        try:
            os.remove(output_name)
        except OSError:
            pass

        if write_csv:
            # write to csv (by appending to file)
            with open(output_name, 'a') as outfile:
                csv_writer = csv.writer(outfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['subdir','frame_num','face_startX','face_startY','face_endX','face_endY',\
                'face_height','face_width','face_x_center','face_y_center','yaw','pitch','roll',\
                'predicted_label','attention_label',])

    def get_file_path(self, frames_dir_path, subdir, frame_num):

        '''

        This function converts frame meta data and converts it into proper
        format and a valid path for loading.

        input param:

        frames_dir_path: string, name of directory containing all frames
        subdir: string, name of a subdirectory for frames within single household
        and hour.
        frame_num: int, frame number

        return: file_path

        '''

        # isolate the frame number
        frame_num_len = len(str(frame_num))
        formatted_frame_num = ''

        # need to append .jpg to frame num, which has either 4 or 5 digits
        if frame_num_len < 5:  # less than 10k frames in subdir, but is 0 padded
            formatted_frame_num = formatted_frame_num = str(frame_num).zfill(4) + '.jpg'
        elif frame_num_len > 4:  # might be 10k+ frames in a subdir, and have 5 digits
            formatted_frame_num = str(frame_num) + '.jpg'

        # create path with full format to find frame
        file_path = '/'.join([frames_dir_path, subdir, formatted_frame_num])
        return file_path

    def correct_data(self, frame_data):

        '''

        This function corrects the data by removing attention scores that are not
        0, 1, or 2.  Also, you can customize this to remove any class, such as the
        1 score.

        '''

        print('[Correcting data (removing improperly labeled samples)]...')

        corrected_data = []

        # iterate through each row in DF
        for index, row in frame_data.iterrows():
            attn_label = int(row[-1])  # isolate the attention label (last entry)

            if attn_label != -1:
                corrected_data.append(row)

        return pd.DataFrame(corrected_data)

    def read_csv(self, file_path):
        print('[Reading in CSV of entire dataset]...')
        frame_data = pd.read_csv(file_path, delim_whitespace=True)
        return frame_data

    # need to take in raw data because we need to know which rows have 'NA'
    def predict_accuracy_on_data(self, frame_directory, X_test_raw, y_test_raw, output_name, one_to_two, write_csv):

        '''

        This function predicts the accuracy of the predictions vs. the annotation.  It reads in the features file
        from train_and_test.py, and predicts using the predictor object.

        '''

        # must iterate one row at a time to set predictions for 'NA' rows (attn == 0)
        print('[Predicting accuracy]...')

        predictor = Attention()

        y_prediction = []
        csv_count = 0
        print('number of samples in this batch', X_test_raw.shape[0])
        for index, row in X_test_raw.iterrows():

            y_pred_single = None
            y_test_single = int(y_test_raw.loc[index][0])  # grab the actual label

            # grab the relevant feature columns for predicting
            X_row = row[['subdir', 'frame_num', 'face_startX','face_startY','face_endX','face_endY','yaw','pitch','roll']]
            new_num_cols = X_row.shape[0]

            # check if valid row
            if self.is_valid_row(X_row[2:]):

                subdir = X_row[0]
                frame_num = X_row[1]
                face_startX = X_row[2]
                face_startY = X_row[3]
                face_endX = X_row[4]
                face_endY = X_row[5]
                yaw = X_row[6]
                pitch = X_row[7]
                roll = X_row[8]

                # need to put face box and head pose in a list of lists to use predictor
                face_box = [[face_startX, face_startY, face_endX, face_endY]]
                head_pose = [[yaw, pitch, roll]]

                X_row = np.array(X_row).reshape(-1,new_num_cols)  # reshape with updated num of cols

                # get file path
                file_path = self.get_file_path(frame_directory, subdir, frame_num)

                # load the frame
                frame = cv2.imread(file_path)

                # predict the attention
                # returns a list, so get the first one
                y_pred_single = predictor.predict_attention(frame, face_box, head_pose)[0]

                # testing if changing 1 to 2s is better
                if one_to_two and y_pred_single == 1:
                    y_pred_single = 2

            else:
                y_pred_single = 0

            y_prediction.append(y_pred_single)  # save the prediction for the row to running list

            # convert full row to list (with string data), appending predicted and actual labels
            predicted_full_entry = row.tolist()
            predicted_full_entry.append(y_pred_single)  # append the prediction
            predicted_full_entry.append(y_test_single)  # append the actual label, 1st entry, convert to int

            # write to csv with the feature vector (with string data) + the predicted label and the actual

            if write_csv:
                self.write_to_csv(predicted_full_entry, output_name)
                csv_count += 1

        print('csv_count for this test house:', csv_count)

        # convert all y_prediction values to a DF for the accuracy report
        y_prediction = pd.DataFrame(y_prediction)

### ----- helper functions

    def write_to_csv(self, row_with_predicted_label, output_name):
        # write to csv here
        with open(output_name, 'a') as outfile:
            csv_writer = csv.writer(outfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(row_with_predicted_label)

    def is_valid_row(self, row):

        for element in row:
            if not math.isnan(element):
                pass
            else:
                return False
        return True

    def split_x_and_y(self, data, one_to_two):
        X_train = data.drop(['attention_label'], axis=1)
        y_train = pd.DataFrame(data['attention_label'])

        # testing if changing 1 to 2s is better
        if one_to_two:
            y_train = y_train.replace(1,2)

        return (X_train, y_train)

    # -------------------  evaluate all results
    def read_labels(self, labels_path):

        actuals = []
        predictions = []
        entry_set = set()
        first_line = True

        with open(labels_path, "r") as f:
            for row in f:
                if first_line:
                    first_line = False
                    continue

                row = list(row.split())
                predictions.append(row[-2])
                actuals.append(row[-1])
        return (actuals, predictions)

    def class_report(self, actuals, predictions):

        print('---------- Report for entire dataset! ----------- \n')

        # check accuracy here
        print('actual shape', len(actuals))
        print('prediction shape', len(predictions))
        print(classification_report(actuals, predictions))
        print(confusion_matrix(actuals, predictions))

def main(frame_directory, output_name_path, csv_results, one_to_two, write_csv):
    test = Hardcode(output_name_path, write_csv)

    frame_data = test.read_csv(csv_results)  # data to train/test
    frame_data = test.correct_data(frame_data)  # remove the incorrect labeled entries

    # preprocess test data, includes normalizing, and adding meta data for csv output
    X_test_raw, y_test_raw = test.split_x_and_y(frame_data, one_to_two)

    # test accuracy with a holdout for validation
    test.predict_accuracy_on_data(frame_directory, X_test_raw, y_test_raw, output_name_path, one_to_two, write_csv)

    # evaluate results for entire data set
    actuals, predictions = test.read_labels(output_name_path)
    # print classification report
    test.class_report(actuals, predictions)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--frame_directory", required=True,
        help="directory of the frames")
    parser.add_argument("-o", "--output_name_path", required=True,
        help="output file path")
    parser.add_argument("-c", "--csv_results", required=True,
        help="path to the results csv")
    parser.add_argument("-t", "--one_to_two", default=False,
        help="converts ones to twos for attention")
    parser.add_argument("-w", "--write_csv", default=False,
        help="writes results to csv")

    args = vars(parser.parse_args())

    # output the predicted labels for each sample
    frame_directory = args['frame_directory']
    output_name_path = args['output_name_path']
    csv_results = args['csv_results']
    one_to_two = args['one_to_two']
    write_csv = args['write_csv']

    main(frame_directory, output_name_path, csv_results, one_to_two, write_csv)
