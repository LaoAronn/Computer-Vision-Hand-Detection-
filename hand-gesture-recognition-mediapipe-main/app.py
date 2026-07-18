#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

WINDOW_NAME = 'Hand Gesture Recognition'

# ---- Gesture menu layout (used by both drawing and click-hit-testing) ----
MENU_X = 10
MENU_Y_START = 150
MENU_HEADER_HEIGHT = 32
MENU_WIDTH = 260
ITEM_HEIGHT = 24
ITEMS_PER_PAGE = 10
ARROW_BTN_SIZE = 28


class MenuState:
    """Shared, mutable state passed into the OpenCV mouse callback.

    OpenCV's mouse callback only fires with (event, x, y, flags, param),
    so we bundle everything the callback needs to read/write into one
    object and pass it in as `param`.
    """

    def __init__(self, labels):
        self.labels = labels
        self.mode = 0
        self.clicked_index = -1
        self.log_feedback_label = ""
        self.log_feedback_frames = 0
        self.page = 0
        self.total_pages = max(1, -(-len(labels) // ITEMS_PER_PAGE))  # ceil div

    def next_page(self):
        self.page = min(self.total_pages - 1, self.page + 1)

    def prev_page(self):
        self.page = max(0, self.page - 1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def mouse_callback(event, x, y, flags, state):
    """Handles clicks on the gesture menu: the < and > page buttons, and
    the up-to-10 gesture rows on the current page.

    Only reacts while in "Logging Key Point" mode (mode == 1), which is
    when the menu is actually drawn on screen.
    """
    if state.mode != 1 or event != cv.EVENT_LBUTTONDOWN:
        return

    header_y = MENU_Y_START - MENU_HEADER_HEIGHT

    # "<" button (previous page), left of the header
    prev_rect = (MENU_X, header_y + 2, MENU_X + ARROW_BTN_SIZE, header_y + 2 + ARROW_BTN_SIZE)
    if prev_rect[0] <= x <= prev_rect[2] and prev_rect[1] <= y <= prev_rect[3]:
        state.prev_page()
        return

    # ">" button (next page), right of the header
    next_x0 = MENU_X + MENU_WIDTH - ARROW_BTN_SIZE
    next_rect = (next_x0, header_y + 2, next_x0 + ARROW_BTN_SIZE, header_y + 2 + ARROW_BTN_SIZE)
    if next_rect[0] <= x <= next_rect[2] and next_rect[1] <= y <= next_rect[3]:
        state.next_page()
        return

    # One of the up-to-10 gesture rows on the current page
    if x < MENU_X or x > MENU_X + MENU_WIDTH or y < MENU_Y_START:
        return
    row = (y - MENU_Y_START) // ITEM_HEIGHT
    if row >= ITEMS_PER_PAGE:
        return
    idx = state.page * ITEMS_PER_PAGE + row
    if 0 <= idx < len(state.labels):
        state.clicked_index = idx


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history 
    finger_gesture_history = deque(maxlen=history_length)

    #  Mode selection 
    mode = 0

    # Gesture menu setup ###################################################
    menu_state = MenuState(keypoint_classifier_labels)
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, mouse_callback, menu_state)

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        if key == ord('['):  # previous menu page
            menu_state.prev_page()
        if key == ord(']'):  # next menu page
            menu_state.next_page()
        number, mode = select_mode(key, mode)

        # A menu click behaves like a single numpad press: it supplies the
        # gesture "number" for this one frame, then is consumed.
        if menu_state.clicked_index != -1:
            number = menu_state.clicked_index
            menu_state.clicked_index = -1

        # Keep the callback aware of the current mode so it only reacts
        # while the menu is actually visible.
        menu_state.mode = mode

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logged = logging_csv(number, mode, pre_processed_landmark_list,
                                     pre_processed_point_history_list,
                                     keypoint_classifier_labels)
                if logged and mode == 1 and 0 <= number < len(keypoint_classifier_labels):
                    menu_state.log_feedback_label = keypoint_classifier_labels[number]
                    menu_state.log_feedback_frames = 20

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Gesture menu overlay (only while in "Logging Key Point" mode) ####
        if mode == 1:
            debug_image = draw_gesture_menu(debug_image, menu_state)

        if menu_state.log_feedback_frames > 0:
            debug_image = draw_log_feedback(debug_image, menu_state.log_feedback_label)
            menu_state.log_feedback_frames -= 1

        # Screen reflection #############################################################
        cv.imshow(WINDOW_NAME, debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9  (still works as a shortcut alongside the menu)
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]

    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def logging_csv(number, mode, landmark_list, point_history_list, keypoint_labels):
    """Returns True if a row was actually written for the keypoint classifier."""
    if mode == 0:
        return False
    if mode == 1 and (0 <= number < len(keypoint_labels)):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        return True
    if mode == 2 and (0 <= number <= 9):
        csv_path = 'model/point_history_classifier/point_history.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list])
    return False


def draw_gesture_menu(image, state):
    """Draws a clickable list of up to 10 gesture labels at a time (from the
    keypoint classifier label CSV), with < and > buttons to move between
    pages. Clicking a row logs one entry for that gesture, the same way
    pressing a numpad key used to."""
    labels = state.labels
    header_y = MENU_Y_START - MENU_HEADER_HEIGHT
    box_height = MENU_HEADER_HEIGHT + ITEMS_PER_PAGE * ITEM_HEIGHT

    overlay = image.copy()
    cv.rectangle(overlay, (MENU_X - 6, header_y - 4),
                 (MENU_X + MENU_WIDTH + 6, header_y + box_height),
                 (40, 40, 40), -1)
    image = cv.addWeighted(overlay, 0.7, image, 0.3, 0)

    # "<" button
    prev_enabled = state.page > 0
    prev_color = (255, 255, 0) if prev_enabled else (110, 110, 110)
    cv.rectangle(image, (MENU_X, header_y + 2),
                 (MENU_X + ARROW_BTN_SIZE, header_y + 2 + ARROW_BTN_SIZE),
                 prev_color, 1)
    cv.putText(image, "<", (MENU_X + 9, header_y + 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, prev_color, 2, cv.LINE_AA)

    # ">" button
    next_enabled = state.page < state.total_pages - 1
    next_color = (255, 255, 0) if next_enabled else (110, 110, 110)
    next_x0 = MENU_X + MENU_WIDTH - ARROW_BTN_SIZE
    cv.rectangle(image, (next_x0, header_y + 2),
                 (next_x0 + ARROW_BTN_SIZE, header_y + 2 + ARROW_BTN_SIZE),
                 next_color, 1)
    cv.putText(image, ">", (next_x0 + 8, header_y + 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, next_color, 2, cv.LINE_AA)

    # Page indicator, centered between the two buttons
    header_text = f"Page {state.page + 1}/{state.total_pages}"
    text_size = cv.getTextSize(header_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    text_x = MENU_X + (MENU_WIDTH - text_size[0]) // 2
    cv.putText(image, header_text, (text_x, header_y + 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    # Up to 10 gesture rows for the current page
    start_idx = state.page * ITEMS_PER_PAGE
    for row in range(ITEMS_PER_PAGE):
        idx = start_idx + row
        if idx >= len(labels):
            break
        item_y = MENU_Y_START + row * ITEM_HEIGHT
        cv.rectangle(image, (MENU_X, item_y), (MENU_X + MENU_WIDTH, item_y + ITEM_HEIGHT - 2),
                     (90, 90, 90), -1)
        cv.rectangle(image, (MENU_X, item_y), (MENU_X + MENU_WIDTH, item_y + ITEM_HEIGHT - 2),
                     (220, 220, 220), 1)
        display_label = labels[idx] if labels[idx] else f"(label {idx})"
        cv.putText(image, f"{idx}: {display_label}", (MENU_X + 6, item_y + 17),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return image


def draw_log_feedback(image, label):
    text = f"Logged: {label}"
    cv.putText(image, text, (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(image, text, (10, 130), cv.FONT_HERSHEY_SIMPLEX, 0.7,
               (0, 255, 0), 1, cv.LINE_AA)
    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index in (0, 1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19):
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index in (4, 8, 12, 16, 20):
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()