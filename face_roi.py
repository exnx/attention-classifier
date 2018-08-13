import numpy as np
import cv2

class FaceROI:

    def create_roi_from_features(self, face_keypoints, frame, frame_size):
        '''

        This function takes in a list of face keypoint coords for a frame and estimates
        a face region of interest (ROI), which can be used itself as the face bounding box.

        face_keypoints: list, with fields as described below
        frame_size: tuple, (heigh,width)

        return: face_box, tuple, a face box / region of interest coordinates (x1,y1,x2,y2)

        # annotations fields key
        # 0 - vatic.video
        # 1 - vatic.frame_id
        # 2 - vatic.bbox_x1
        # 3 - vatic.bbox_y1
        # 4 - vatic.bbox_x2
        # 5 - vatic.bbox_y2
        # 6 - reye_x
        # 7 - reye_y
        # 8 - leye_x
        # 9 - leye_y
        # 10 - rear_x
        # 11 - rear_y
        # 12 - lear_x
        # 13 - lear_y
        # 14 - nose_x
        # 15 - nose_y
        # 16 - neck_x
        # 17 - neck_y
        # 18 - attention

        '''

        # print('[Creating face bounding box from FaceBoxer]...')

        # increases size of the face box (otherwise it's too tight)
        face_box_multiplier = 1.4  # between 1.4 and 1.6 seems to be ok
        min_height = 80
        min_width = 80

        # reye_x, reye_y, leye_x, leye_y, rear_x, rear_y, nose_x, nose_y, neck_x, neck_y
        width = frame_size[1]
        height = frame_size[0]

        # un-normalize coordinates by mult. by frame size
        reye_x = float(face_keypoints[0]) * width
        reye_y = float(face_keypoints[1]) * height
        leye_x = float(face_keypoints[2]) * width
        leye_y = float(face_keypoints[3]) * height
        rear_x = float(face_keypoints[4]) * width
        rear_y = float(face_keypoints[5]) * height
        lear_x = float(face_keypoints[6]) * width
        lear_y = float(face_keypoints[7]) * height
        nose_x = float(face_keypoints[8]) * width
        nose_y = float(face_keypoints[9]) * height
        neck_x = float(face_keypoints[10]) * width
        neck_y = float(face_keypoints[11]) * height

        # consolidate neck and nose coords
        neck_coords = [(neck_x,neck_y)]
        nose_coords = [nose_x, nose_y]

        # only include face coordinate if the coord is > 0 (which means it's found)
        face_coords = []
        if lear_x > 0:
            face_coords.append((lear_x,lear_y))
            # cv2.circle(frame, (int(lear_x),int(lear_y)), 3, (0,0,255), -1)
        if leye_x > 0:
            face_coords.append((leye_x, leye_y))
            # cv2.circle(frame, (int(leye_x),int(leye_y)), 3, (0,0,255), -1)
        if nose_x > 0:
            face_coords.append((nose_x, nose_y))
            # cv2.circle(frame, (int(nose_x), int(nose_y)), 3, (0,0,255), -1)
        if reye_x > 0:
            face_coords.append((reye_x,reye_y))
            # cv2.circle(frame, (int(reye_x),int(reye_y)), 3, (0,0,255), -1)
        if rear_x > 0:
            face_coords.append((rear_x, rear_y))
            # cv2.circle(frame, (int(rear_x), int(rear_y)), 3, (0,0,255), -1)

        face_box = None  # for return value

        # if at least the nose if found we calc. a face box
        if nose_x > 0:

            # normalize coordinates
            neck_coords = np.array(neck_coords)
            face_coords = np.vstack(face_coords)
            nose_coords = np.array(nose_coords)

            # width of face (not box)
            face_w = np.max(np.sqrt(np.sum(np.power(face_coords - nose_coords, 2), axis=1)))

            # declare face box coords
            x1 = None
            y1 = None
            x2 = None
            y2 = None

            # if nose and neck found use this
            if nose_x > 0 and neck_x > 0:
                # print('-----face from nose and neck points')

                # height of face (not box)
                face_h = np.sqrt(np.sum(np.power(neck_coords - nose_coords, 2)))

                # get various face ratios
                h = (neck_coords - nose_coords) / face_h * max(face_w, face_h * 0.5)
                chin_coords = h + nose_coords
                forhead_coords = -h + nose_coords

                all_coords = np.vstack([face_coords, chin_coords, forhead_coords])

                # get the least and max coords to be the initial face boundary box
                face_min = np.min(all_coords, axis=0)
                face_max = np.max(all_coords, axis=0)
                face_bbox = np.array([face_min, face_max])

                # increase size of bbox
                face_mean = np.mean(face_bbox, axis=0)
                face_bbox = face_box_multiplier * (face_bbox - face_mean) + face_mean

                # convert to x1,y1 format and int
                x1 = int(face_bbox[0][0])
                y1 = int(face_bbox[0][1])
                x2 = int(face_bbox[1][0])
                y2 = int(face_bbox[1][1])

                # centers the uneven box around the nose
                x1, y1, x2, y2 = self.center_box(x1,y1,x2,y2,frame_size)

            # if nose found only (and no neck found) use this instead
            # if the nose is not found, then we disregard face detection for the purpose
            # of attention
            else:

                # get max euclidian distance between nose and other face coord
                euclid_dist = face_w
                chin_coords = euclid_dist + nose_coords
                forhead_coords = -euclid_dist + nose_coords
                all_coords = np.vstack([face_coords, chin_coords, forhead_coords])

                face_min = np.min(face_coords, axis=0)
                face_max = np.max(face_coords, axis=0)
                face_bbox = np.array([face_min, face_max])

                # increase size of bbox
                face_mean = np.mean(face_bbox, axis=0)
                face_bbox = face_box_multiplier * (face_bbox - face_mean) + face_mean

                # convert to x1,y1 format and int
                x1 = int(face_bbox[0][0])
                y1 = int(face_bbox[0][1])
                x2 = int(face_bbox[1][0])
                y2 = int(face_bbox[1][1])

                # estimator is too short, need to lengthen
                x1, y1, x2, y2 = self.lengthen_box(x1,y1,x2,y2,frame_size)

            curr_height = y2-y1
            curr_width = x2-x1

            face_box = (x1, y1, x2, y2)  # store into tuple
            # print('Current --- height {}, width {}:'.format(curr_height,curr_width))

            # set a min face ROI no matter what for tiny face to find a face
            if curr_height < min_height or curr_width < min_width:
                face_box = self.create_min_box(x1, y1, x2, y2, min_height, min_width)
                x1, y1, x2, y2 = face_box
                new_height = y2-y1
                new_width = x2-x1
                # print('New --- height {}, width {}:'.format(new_height,new_width))

            # need to make sure all coordinates are in bound first, and are ints
            # but only if nose coordinate was found
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, width)
            y2 = min(y2, height)
            face_box = (x1,y1,x2,y2)

        return face_box  # return tuple

    def create_min_box(self, x1, y1, x2, y2, min_height, min_width):

        '''

        This function ensures a min. face ROI.

        '''

        # print('create min box called')

        center_x = (x2-x1)/2
        center_y = (y2-y1)/2

        y1 = y1 - min_height/2
        y2 = y2 + min_height/2

        x1 = x1 - min_width/2
        x2 = x2 + min_width/2

        return (int(x1), int(y1), int(x2), int(y2))

    def lengthen_box(self,x1,y1,x2,y2,frame_size):

        '''
        This function lengthens the height of a face box, which is called when
        only the nose coordinate is found.

        x1, y1, x2, y2: float, coordinates of initial face box
        frame_size: tuple, (height, width)

        return: tuple, (x1, y1, x2, y2) coords of face box

        '''

        full_frame_width = frame_size[1]
        full_frame_height = frame_size[0]

        width = x2 - x1
        curr_height = y2 - y1
        hei_wid_diff = abs(curr_height-width)
        # add the difference to top and bottom equally
        y1 = int(y1 - hei_wid_diff/2)
        y2 = int(y2 + hei_wid_diff/2)

        return (x1, y1, x2, y2)

    def center_box(self, face_startX, face_startY, face_endX, face_endY, frame_size):
        '''

        This function is used to center an initial face bounding box when a nose
        and neck coord is found.  It is used when the face box is too "tall",
        and we need a more square box.

        face_startX, face_startY, face_endX, face_endY: float, coordinates for face box
        frame_size: tuple, (height, width)

        return: tuple, (face_startX, face_startY, face_endX, face_endY) coords of face box

        '''

        full_frame_width = frame_size[1]
        full_frame_height = frame_size[0]

        # check difference between height and width
        face_width = face_endX - face_startX
        curr_height = face_endY - face_startY
        hei_wid_diff = curr_height - face_width

        # if box is too tall
        if hei_wid_diff > 0:
            # reduce height from top and bottom portions of box
            face_startY = int(face_startY + hei_wid_diff/2)
            face_endY = int(face_endY - hei_wid_diff/2)

        return (face_startX, face_startY, face_endX, face_endY)
