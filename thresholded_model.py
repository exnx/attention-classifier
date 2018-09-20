import cv2
import numpy as np


class Attention:

    def predict_attention(self, frame, face_boxes, head_poses):

        '''

        This module predicts the attention for a list of faces in a frame.  It
        is built on thresholds based on where the person is in the room and an
        estimated depth (forward distance from the camera).

        It models a room into discrete sections in a room in the x, y and z directions.

        X is modeled in the left to right directions, Y in the up an down, and
        Z as the depth into the image.

        X has 3 sections, left, center and right.
        Y has 3 sections, top, middle and bottom.
        Z has 3 sections, close, avg and far.

        Depth is estimated by anecdotal calibration of a subset of frames,
        and by pixel size of a face.

        input param:

        frame:
        face_boxes:
        head_poses:

        return: predictions, list, of attention scores for each face in frame


        '''

        num_faces = len(face_boxes)

        predictions = []

        for face_num in range(num_faces):

            attention_pred = None
            # set curr face and pose with face_num position
            curr_face_box = face_boxes[face_num]
            curr_head_pose = head_poses[face_num]

            # if either of these are None then predict a 0
            if not curr_face_box or not curr_head_pose:
                attention_pred = 0

            # otherwise use head pose to estimate attention
            else:

                # get the face box and head poses
                startX, startY, endX, endY = curr_face_box
                face_height = endY - startY
                face_width = endX - startX
                tx = int(face_width/2 + startX)  # translation x of face center
                ty = int(face_height/2 + startY) # translation y of face center
                yaw, pitch, roll = curr_head_pose

                depth_est = 0 # in ft

                # use face box height (in pixels) to estimate depth
                if face_height < 30:
                    depth_est = 22  # in ft
                elif face_height < 50:
                    depth_est = 18
                elif face_height < 70:
                    depth_est = 14
                elif face_height < 100:
                    depth_est = 10
                elif face_height < 180:
                    depth_est = 7
                else:
                    depth_est = 4

                focal_length = 525  # in pixels

                # set boundaries for each section in the room based on 1080x1920 image size

                # boundaries in the x direction
                left_bound_x = 540
                center_bound_x = 1380
                right_bound_x = 1920

                # boundaries in the y direction
                top_bound_y = 360
                middle_bound_y = 850
                bottom_bound_y = 1080

                # z depth bounds
                close_bound_z = 8  # 0 - 6 ft close depth
                avg_bound_z = 15  # avg depth
                far_bound_z = 21  # far depth

                # declare initial position
                room_loc_x = None
                room_loc_y = None
                room_loc_z = None

                # 
                x_loc_offset = 0
                y_loc_offset = 0
                z_loc_offset = 0
                z_loc_pitch_offset = 0

                x_upper_offset = 0
                x_lower_offset = 0

                # in x direction
                if tx < left_bound_x:
                    room_loc_x = 'left'
                    # x_loc_offset = 15

                    x_upper_offset = -15
                    x_lower_offset = -15

                elif tx < center_bound_x:
                    room_loc_x = 'center'
                    # x_loc_offset = 0

                    x_upper_offset = 0
                    x_lower_offset = 0

                elif tx < right_bound_x:
                    room_loc_x = 'right'
                    # x_loc_offset = -15

                    x_upper_offset = 15
                    x_lower_offset = 15

                # in z direction, changes yaw range tolerance slightly
                if depth_est < close_bound_z:
                    room_loc_z = 'close'
                    z_loc_offset = 5
                    z_loc_pitch_offset = 5

                elif depth_est < avg_bound_z:
                    room_loc_z = 'avg'
                    z_loc_offset = 0
                    z_loc_pitch_offset = 0

                # range for attention decreases farther you are away
                elif depth_est < far_bound_z:
                    room_loc_z = 'far'
                    z_loc_offset = -5
                    z_loc_pitch_offset = -5

                # in y direction, changes pitch range tolerance slightly
                if ty < top_bound_y:
                    room_loc_y = 'top'
                    # y_loc_offset = 3

                elif ty < middle_bound_y:
                    room_loc_y = 'middle'
                    y_loc_offset = 0

                elif ty < bottom_bound_y:
                    room_loc_y = 'bottom'

                    # likely because camera is often tilted up too high (positive pitch)
                    if room_loc_z == 'close':
                        y_loc_offset = 15

                set_attention = False

                # attention 2 ranges
                yaw_range_2 = [-20 + x_lower_offset - z_loc_offset, 20 + x_upper_offset + z_loc_offset]
                pitch_range_2 = [-10 - y_loc_offset - z_loc_pitch_offset, 30 + y_loc_offset + z_loc_pitch_offset]
                roll_range_2 = [-25, 25]

                # attention 1 ranges
                yaw_range_1 = [-25 + x_lower_offset - z_loc_offset, 25 + x_upper_offset + z_loc_offset]
                pitch_range_1 = [-15 - y_loc_offset - z_loc_pitch_offset, 35 + y_loc_offset + z_loc_pitch_offset]
                roll_range_1 = [-32, 32]

                if yaw >= yaw_range_2[0] and yaw <= yaw_range_2[1] and pitch >= pitch_range_2[0] and \
                    pitch <= pitch_range_2[1] and roll >= roll_range_2[0] and roll <= roll_range_2[1]:

                    attention_pred = 2

                    if room_loc_z == 'far':
                        attention_pred = 1

                    set_attention = True

                elif not set_attention and yaw >= yaw_range_1[0] and yaw <= yaw_range_1[1] and pitch >= pitch_range_1[0] \
                    and pitch <= pitch_range_1[1] and roll >= roll_range_1[0] and roll <= roll_range_1[1]:

                    attention_pred = 1
                    set_attention = True
                    #
                    # if room_loc_z == 'far':
                    #     attention_pred = 0

                else:
                    attention_pred = 0

            predictions.append(attention_pred)

        return predictions
