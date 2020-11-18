import matplotlib
# matplotlib.use('TkAgg')

import cv2
import numpy as np
import ctypes
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from vision_helper import (
    show_object_near_space,
    plt_to_img
)

# Viewer for a general camera Observer object
class SincDataViewer(object):
    out = None  # variable holding writer object for saving a view

    def __init__(self, window_name='cam_0', data_folder='./../data/', detector=None):
        self.window_name = window_name
        self.detector = detector
        self.data_folder = data_folder

        # screen size
        self.screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        self.x_coord = np.int(self.screen_size[0]/3)

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        if self.out is not None:
            self.out.release()
        cv2.destroyAllWindows()

    def show_frame(self, step, snapshot):
        raise NotImplementedError


# Viewer for a Scholar (learning) object
class LearningViewer(object):
    out = None  # variable holding writer object for saving view

    def __init__(self, pos_arg='right', name='scholar1', scholar=None, data_folder='./../data/'):
        self.window_name = name
        self.scholar = scholar

        # screen size
        self.screen_size = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
        if pos_arg == 'left':
            self.x_coord = 0
        elif pos_arg == 'center':
            self.x_coord = np.int(self.screen_size[0]/3)
        elif pos_arg == 'right':
            self.x_coord = np.int(self.screen_size[0] / 3) * 2
        else:
            raise ValueError("Please enter the correct screen position value (i.e. one of: 'left', 'center' or 'right')")

        self.frame_width = int(self.screen_size[0] / 3)
        self.frame_height = int(self.screen_size[1] / 3)

        cv2.startWindowThread()

    # should be called at the end
    def release(self):
        cv2.destroyAllWindows()

    def show_frames(self, train_error=None, train_accuracy=None):

        learing_figure = plt.figure()
        axes = learing_figure.add_subplot(111)

        axes.set_autoscale_on(True)
        axes.autoscale_view(True, True, True)

        learning_curve, = plt.plot([0, 1, 2, 3], [0, 1, 2, 3], 'ro-')

        width_to_display = int(np.floor((350 / self.frame_height) * self.frame_width))
        height_to_display = int(self.frame_height)

        # show detection
        if train_error is not None and train_accuracy is not None:
            # learning_curve.set_ydata(train_error)
            learning_curve.set_data(np.arange(len(train_error)), train_error)
            axes.relim()
            axes.autoscale_view(True, True, True)
            learing_figure.canvas.draw()
            # learning_curve_img = plt_to_img(self.learing_figure)

            learning_curve_img = np.fromstring(learing_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            learning_curve_img = learning_curve_img.reshape(learing_figure.canvas.get_width_height()[::-1] + (3,))

            # print the image in the data at step, unless an image was passed to this function
            height_to_display = int(np.floor((learning_curve_img.shape[0] / width_to_display) * self.frame_height))
            reshaped_roi = cv2.resize(learning_curve_img, (width_to_display, height_to_display))

            cv2.imshow(self.window_name + 'learning_curve', reshaped_roi)
            cv2.moveWindow(self.window_name + 'learning_curve', self.x_coord, np.int(self.frame_height + 40))

        cv2.waitKey(1)
        # time.sleep(3)
        plt.close('all')
        return True


class GameOfLifeViewer(SincDataViewer):

    def __init__(self, board_size, save_video=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        screed_div_factor = 3

        # self.frame_width = board_size[1]
        self.frame_width = int(self.screen_size[0] / screed_div_factor)
        self.frame_height = self.frame_width*board_size[1]/board_size[0]

        # Define the codec and create VideoWriter object
        self.save_video = save_video
        if self.save_video:
            self.out = cv2.VideoWriter(
                self.data_folder+'out_{}.avi'.format(self.window_name),
                cv2.VideoWriter_fourcc(*'XVID'),
                25,  # todo: parameter must be based on time interval (sampling period)
                (np.int32(self.detector.camera.get(3)), np.int32(self.detector.camera.get(4))),
                True
            )

    def show_frame(self, step, board=None, text=""):
        roi = board.copy()

        # print the image in the data at step, unless an image was passed to this function
        width_to_display = int(self.frame_width)
        height_to_display = int(self.frame_height)
        reshaped_roi = cv2.resize(roi, (width_to_display, height_to_display), cv2.INTER_NEAREST)

        # show gol board
        cv2.putText(
            reshaped_roi,
            text,
            (20, 20),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=.7,
            color=(255, 0, 0),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        # save snapshot to video
        if self.save_video:
            self.out.write(reshaped_roi)

        cv2.imshow(self.window_name, reshaped_roi)
        cv2.moveWindow(self.window_name, self.x_coord, 0)

        print(text)

        cv2.waitKey(1)
        return True


class ParityViewer(SincDataViewer):

    def __init__(self, save_video=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        screed_div_factor = 3

        # self.frame_width = board_size[1]
        self.frame_width = int(self.screen_size[0] / screed_div_factor)

        # Define the codec and create VideoWriter object
        self.save_video = save_video
        if self.save_video:
            self.out = cv2.VideoWriter(
                self.data_folder+'out_{}.avi'.format(self.window_name),
                cv2.VideoWriter_fourcc(*'XVID'),
                25,  # todo: parameter must be based on time interval (sampling period)
                (np.int32(self.detector.camera.get(3)), np.int32(self.detector.camera.get(4))),
                True
            )

    def show_frame(self, step, bytes_string=None, results=None):

        roi = bytes_string.copy()

        self.frame_height = self.frame_width*bytes_string.shape[0]/bytes_string.shape[1]

        # print the image in the data at step, unless an image was passed to this function
        width_to_display = int(self.frame_width*1.2)

        learing_figure = plt.figure()
        axes = learing_figure.add_subplot(111)

        axes.set_autoscale_on(True)
        axes.autoscale_view(True, True, True)

        # create matrix for byte visualization
        axes.imshow(roi, interpolation='nearest', cmap='Greys')
        axes.set(xticks=np.arange(roi.shape[1]),
               yticks=np.arange(roi.shape[0]),
               ylabel='bytes (rows)',
               xlabel='bits (row-elements)')

        # Loop over data dimensions and create text annotations (bits).
        fmt = 'd'
        thresh = 1 / 2.
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                axes.text(j, i, format(int(roi[i, j]), fmt),
                        ha="center", va="center",
                        color="white" if roi[i, j] > thresh else "black")

        # red rectangles for mistakes, green for correct answers and blue for currently processed byte
        for i in range(bytes_string.shape[0]-1, int(step-1), -1):
            if results[i] == 0:
                edgecolor = 'r'
            else:
                edgecolor = 'g'
            rect = patches.Rectangle((0-.4, i - .4), bytes_string.shape[1]-.2, .8, linewidth=3, edgecolor=edgecolor , facecolor='none')
            axes.add_patch(rect)
        if step > 1:
            rect = patches.Rectangle((0 - .4, int(step) - 1.4), bytes_string.shape[1]-.2, .8, linewidth=3, edgecolor='b', facecolor='none')
            axes.add_patch(rect)

        learing_figure.tight_layout()

        # transform plt figure intow cv2 image
        axes.relim()
        axes.autoscale_view(True, True, True)
        learing_figure.canvas.draw()
        img = np.fromstring(learing_figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(learing_figure.canvas.get_width_height()[::-1] + (3,))

        # create proper roi
        height_to_display = int(np.floor((img.shape[0] / width_to_display) * self.frame_height))
        reshaped_roi = cv2.cvtColor(cv2.resize(img, (width_to_display, height_to_display)), cv2.COLOR_RGB2BGR)

        # save snapshot to video
        if self.save_video:
            self.out.write(reshaped_roi)

        # show roi
        cv2.imshow(self.window_name, reshaped_roi)
        cv2.moveWindow(self.window_name, self.x_coord, 0)

        cv2.waitKey(1)

        # slow down for visualization
        time.sleep(.2)
        return True

