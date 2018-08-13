import csv

class Annotation:

    def __init__(self, annotations_path, frame_size):
        # reading in annotations csv, and storing all entries into a list
        self.frame_size = frame_size

    def get_annotation_rows(self, annotations_path):

        '''

        Given a annotation file in csv, this function will return the all the rows

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

        annotation_rows = []

        with open(annotations_path, "rt") as file:
            # reader is all the rows loaded into memory at the beginning
            reader = csv.reader(file, delimiter=',')
            counter = 0  # keep a counter of entries
            next(reader)  # skip the header

            # iterate through rows
            for row in reader:
                if row:  # only take care of empty rows
                    counter += 1  # increment counter
                    row[0] = row[0].replace('/','-')  # replace to fit format of frame jpegs

                    # strip white space from each entry
                    new_row = [entry.strip() for entry in row]

                    annotation_rows.append(new_row)

        return annotation_rows
