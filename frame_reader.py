import os
import cv2

class FrameReader:

    def create_frames_set(self, frames_dir_path):

        '''

        Creates a set of all the frames in a directory.  Allows for fast checking
        if an annotation path matches with photos in the frame directory.

        '''

        exclude_prefixes = ('__', '.')  # exclusion prefixes
        frames_to_test = []  # will store all file names (and full path) here

        # walk through the directory and find all the files
        for root, dirs, files in os.walk(frames_dir_path):
            for name in files:
                if not name.startswith(exclude_prefixes):  # exclude hidden folders
                    # need to add append to list first
                    frames_to_test.append(os.path.join(root, name))

        # filter correct rows and put in a set instead
        frames_set = set()

        for file_path in frames_to_test:
            # need these for saving to csv
            root = file_path.split('/')[-3]  # just the root dir
            subdir = file_path.split('/')[-2]  # just the household video dir
            frame_num = file_path.split('/')[-1]  # just the frame file name

            # filtering out the right file types
            if subdir[:2] == 'HH' and frame_num[-3:] == 'jpg':

                # convert format
                frame_num = frame_num[:-4]
                frame_num = str(int(frame_num))
                relative_path = subdir + '/' + frame_num
                frames_set.add(relative_path)
        return frames_set

    def load_frame(self, frame_path):

        '''

        Loads a frame give a file path if it exists

        '''

        root = frame_path.split('/')[-3]  # just the root dir
        subdir = frame_path.split('/')[-2]  # just the household video dir
        frame_num = frame_path.split('/')[-1]  # just the frame file name

        # make sure that this directory is a subdir with HH prefix (for testing)
        # and check if file is a .jpg file
        if subdir[:2] == 'HH' and frame_num[-3:] == 'jpg':
            frame = cv2.imread(frame_path) # Read the image with OpenCV
            return frame
