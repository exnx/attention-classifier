import numpy as np
import cv2
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import headpose.hopenet as headpose
import headpose.utils as utils

class HopeHeadPose:

    def __init__(self):

        cudnn.enabled = False

        batch_size = 1
        snapshot_path = './headpose/hopenet_robust_alpha1.pkl'

        # ResNet50 structure
        self.model = headpose.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        print('Loading snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(snapshot_path, map_location='cpu')
        self.model.load_state_dict(saved_state_dict)

        print('Loading data.')

        self.transformations = transforms.Compose([transforms.Scale(224),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        print('Ready to test network.')

        # Test the Model
        self.model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor)

    def get_pose(self, frame, face_bounding_box, display_frame=False):

        frame_height = frame.shape[0]  # get shape
        frame_width = frame.shape[1]

        startX = face_bounding_box[0]  # get coord of face
        startY = face_bounding_box[1]
        endX = face_bounding_box[2]
        endY = face_bounding_box[3]

        bbox_height = int(endY - startY)  # get ratios for face
        bbox_width = int(endX - startX)

        yaw_predicted = 'NA'  # declare head pose vars
        pitch_predicted = 'NA'
        roll_predicted = 'NA'

        # check to make sure that the BB is approx square, and is positive
        # and all in bounds
        if bbox_width >= 0 and bbox_height >= 0 and bbox_height/bbox_width > 0.5 \
            and startX >= 0 and startY >= 0 and endX <= frame_width and endY <= frame_height:

            # Crop image
            img = frame[startY:endY,startX:endX]
            img = Image.fromarray(img)

            # Transform (resizes and centers the image)
            img = self.transformations(img)
            img_shape = img.size()

            # adds a dimension for number of frames in the first dimension
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])

            img = Variable(img)  # convert to pytorch variable

            yaw, pitch, roll = self.model(img)  # pass through the CNN

            # returns a tensor that needs to be converted to a classification
            yaw_predicted = F.softmax(yaw)
            pitch_predicted = F.softmax(pitch)
            roll_predicted = F.softmax(roll)

            # Get continuous predictions in degrees.
            yaw_predicted = int(torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99)
            pitch_predicted = int(torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99)
            roll_predicted = int(torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99)

            # will draw the axis on the frame
            if display_frame:
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, \
                    tdx = (startX + endX) / 2, tdy= (startY + endY) / 2, size = bbox_height/2)

        return (yaw_predicted, pitch_predicted, roll_predicted)
