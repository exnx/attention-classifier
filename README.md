# Attention classifier using head pose.

TLDR;

run **train_and_test.py** on a directory of household images and their set of annotations.

---------

This model classifies the level of viewing attention for a person in a living room.
The attention classifications are 0, 1, and 2, where 0 is not paying attention,
1 is likely paying attention, and 2 is definitely paying attention.  The model can
also be set to a binary classification as well, as 0 and 2.

The model extracts and uses convolutional neural networks (CNNs) to get the location
of the face in the frame, and the angles of the head (head pose) as features into the
attention prediction.  A pipeline of tasks are created for the system.

**Requirements:**

It's recommended to use a conda virtual environment.  You can use the requirements.txt
file (conda install --yes --file requirements.txt), which include TensorFlow, PyTorch,
OpenCV, Numpy, pandas, sklearn, imutils, pickle, scipy, glob, pylab, PIL, and a few others.

**Model pipeline tasks:**
    face ROI -> face bounding box -> head pose -> attention score

**Model modules/scripts: for tasks:**
    face_roi.py -> tiny_face_detect.py -> head_pose.py -> thresholded_model.py

**Pipeline description**

**1.  Face ROI:**  The first step is to create the face region of interest (ROI).  
This model currently assumes facial features from PAF to create the face ROI.
These are the coordinates for the eyes, ears, nose and neck.  This
step may not be needed in the future if a better face detector is used.

The **face_roi.py** script takes in a list of facial features for each face in a frame.
This narrows the search area for face detection for the actual face detection.

**2.  Face bounding box:**  Next, the script **tiny_face_detect.py** in the **TinyFace** module is used
to find a tighter fitting face bounding box.  The tighter bounding box significantly
improves the accuracy of the head pose model (next step).  It takes in a list of face ROIs
in frame.

**3.  Head pose:**  The script **head_pose.py** in the **headpose** module is used to find the angle
 of the face given as yaw, pitch and roll.  These represent angles in the x (horizontal),
y (vertical), and z (into the image) axis, respectively.  It takes in a list of face
boxes bounding for each face in a frame.

**4.  Attention prediction:**  The **thresholded_model.py** script uses a simple thresholding
to classify angles where a person is estimated to be paying attention to the screen
in front of them.  The camera is assumed to be placed in the very center of the camera image.
This predictor uses the face location in the room, for each x, y, z directions,
and the angle of the head.  It takes in a list of face boxes and head poses in a frame.

Note: each of these tasks are done by faces in a image frame.  Each module accepts
**lists** of ROIs, bounding boxes, and head poses, representing annotations for each
face in a single frame.

#Using the modules:#
Any modules can be swapped out or used independently.  For specific tasks, directions
are provided below.

**1. Train and Test:**  to train and test on a set of images/frames,
the **train_and_test.py** script is provided. The main purpose of this script is
to extract all face bounding boxes and head poses **features**.  This will take
significant amount of time, hours, at ~1 sec / frame.

By default, a thresholded model is also used to predict the attention score.  
This does not need to be trained as it takes in the features and simple thresholds
values to predict attention.  Evaluation results are also automatically done at
the end of feature extraction.

To run, pass in the following arguments (change args as necessary):

python train_and_test.py --test_frames_path test-frames --annotations_path Inderbir-annotations.csv --output_training_name output_training_data_test.csv --display_frame True --write_csv False

Note: this script takes many hours, a day or more on large data sets. (~1 sec/frame)

After extracting all the features from the frames, you need to decide which predictor to
use. The most complete is the thresholded model, which thresholds angles based on a person's
position in the room to predict attention.

To run the train_and_test.py script, you will need a directory of frames, and a csv file of the
annotations for the facial features and true attention labels.  It uses **frame_reader.py**
and **annotation_reader.py** scripts to traverse a directory and read a csv file.  See these
scripts and the 'test-frames directory' and the 'Inderbir-annotations.csv' files for the
format structure of the input.

Options to display the images with select annotations are also given in this script.

When providing a csv of annotations, the following fields are used.  If you alter the
features or rearrange the order, you will need to change the face_roi.py file correspondingly
to read the appropriate facial features from your csv annotation file.  Alternatively,
you can directly feed in face bounding boxes to the head pose script, and by pass
the face ROI and face bounding box creation.

Annotation fields used in **face_roi.py**:

0 - vatic.video

1 - vatic.frame_id

2 - vatic.bbox_x1  # not needed in this model

3 - vatic.bbox_y1  # not needed in this model

4 - vatic.bbox_x2  # not needed in this model

5 - vatic.bbox_y2  # not needed in this model

6 - reye_x

7 - reye_y

8 - leye_x

9 - leye_y

10 - rear_x

11 - rear_y

12 - lear_x

13 - lear_y

14 - nose_x

15 - nose_y

16 - neck_x

17 - neck_y

18 - attention


**2. Training: optional**

Skip this step if using the threshold_model.py script.

There are trained models using an SVM and random forest classifier based on the
output features of the train_and_test.py script.  

See the **machine-learning-trainers** to train a predictor on the features
instead of the thresholded coded model.  Note, these models are less documented and ready
to use and need modifying (paths to features file, for example).  They did not provide
good accuracy. However, they may provide a good basis code for cross validating and
for testing other machine learning models.

These scripts will also test the training and validation data.

**3. Test results evaluation:  optional**  

After feature extraction from the train_and_test.py script, you can evaluate the overall accuracy
of the thresholded code model predictor by using the **test-thresholded-model.py** script.
This may be redundant as the train_and_test.py script evaluates the accuracy, but this
test-thresholded-model.py lets you fine tune the classification, such as convert 1s to 2s.
You can also swap out the classified and use this script to test other machine learning models,
ie, random forest, SVM.

To run, pass in the following arguments (change args as necessary):

python test-thresholded-model.py --frame_directory test-frames --output_name_path predicted_labels_hardcode_rev2.csv --csv_results output_features_aug7.csv --one_to_two False --write_csv True

Results directory has the output of some test evaluations for the thresholded coded,
SVM and random forest models.  These include features and the predicted and
annotated labels.
