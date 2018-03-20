#!/usr/bin/python
from tkinter import *
from multiprocessing import Process, Queue
import time


class SVSVisualizer(Process):

    def __init__(self, state=None):
        Process.__init__(self)
        self.queue = Queue()
        if state is not None:
            self.put_state(state)

    def run(self):
        self.master = Tk()
        self.panel_width = 500
        self.panel_height = 500
        self.w = Canvas(self.master, width=self.panel_width, height=self.panel_height)
        self.w.pack()
        self._animate()

    def put_state(self, state):
        self.queue.put(state)

    def _animate(self):
        self.master.after(100, self._check_queue())  # blocking
        self.master.mainloop()

    def _check_queue(self):
        if not self.queue.empty():
            obj = self.queue.get_nowait()
            if str(obj) == 'close':
                print("close")
                self.master.quit()
                self.terminate()
            self._render_state(obj)
        self.master.after(100, self._check_queue)

    def close(self):
        self.queue.put('close')

    def _render_state(self, state):
        self.w.delete("all")
        # Scale room dimensions
        room_width = state.room_width
        room_height = state.room_height

        panel_width = self.panel_width - 100
        panel_height = self.panel_height - 100
        panel_ratio = panel_height/panel_width
        aspect_ratio = room_height/room_width

        if panel_ratio > aspect_ratio:
            target_width = panel_width
            target_height = panel_width * aspect_ratio
        else:
            target_height = panel_height
            target_width = panel_height/aspect_ratio

        scale_factor_width = target_width / room_width
        scale_factor_height = target_height / room_height

        # build room (rectangle: x1,y1,x2,y2)
        self.w.create_rectangle(50, 50, 50 + target_width, 50 + target_height, fill="#ffffff")
        # build and colorize room segments, if defined
        if state.room_segmentation is not None:
            min_room_segments = []
            min_dist = 1
            for segment in state.room_segmentation.segments:
                if segment.value == min_dist:
                    min_room_segments.append(segment)
                elif segment.value < min_dist:
                    min_room_segments.clear()
                    min_room_segments.append(segment)
                    min_dist = segment.value
                scaled_transformed_x = 50 + segment.left_bottom[0] * scale_factor_width
                scaled_transformed_y = 50 + segment.left_bottom[1] * scale_factor_height
                color = '#%02x%02x%02x' % rgb(state.room_segmentation.min_value, state.room_segmentation.max_value,
                                              segment.value)
                self.w.create_rectangle(scaled_transformed_x, scaled_transformed_y,
                                        scaled_transformed_x+scale_factor_width,
                                        scaled_transformed_y+scale_factor_height, fill=color, width=1)
        # build obstacle (rectangle)
        self.w.create_rectangle(50 + state.obstacle_left_bottom[0]*scale_factor_width,
                                50 + state.obstacle_left_bottom[1]*scale_factor_height,
                                50 + (state.obstacle_left_bottom[0] + state.obstacle_width) * scale_factor_width,
                                50 + (state.obstacle_left_bottom[1] + state.obstacle_height) * scale_factor_height,
                                fill="lightgrey")
        # build robot (circle)
        robot_position = (50 + (state.robot_position[0]+0.5)*scale_factor_width,
                          50+(state.robot_position[1]+0.5)*scale_factor_height)
        robot_radius = 0.5 * scale_factor_width
        self.w.create_oval(robot_position[0] - robot_radius, robot_position[1] - robot_radius,
                           robot_position[0] + robot_radius, robot_position[1] + robot_radius,
                           fill='#ffffff')
        # build dust (points)
        dust_radius = 0.2 * scale_factor_width
        for dust in state.dirt_positions:
            dust_position = (50+(dust[0]+0.5)*scale_factor_width, 50+(dust[1]+0.5)*scale_factor_height)
            self.w.create_oval(dust_position[0] - dust_radius, dust_position[1] - dust_radius,
                               dust_position[0] + dust_radius, dust_position[1] + dust_radius,
                               fill="#000000")


def rgb(minimum, maximum, value):
    minimum, maximum = float(minimum)+sys.float_info.min, float(maximum)
    ratio = 2 * (value-minimum) / (maximum - minimum)
    r = int(max(0, 255*(1 - ratio)))  # switch r and b for inverting color scale
    b = int(max(0, 255*(ratio - 1)))
    g = 255 - b - r
    return r, g, b


if __name__ == '__main__':
    """
    print exemplary state.
    """
    from case_study_svs.cfm import State
    state = State(room_height=10, room_width=10, obstacle_left_bottom=(3, 4), obstacle_width=4, obstacle_height=2,
                      dirt_position=(2, 2), robot_position=(8, 8))
    visualizer = SVSVisualizer()
    visualizer.start()
    visualizer.put_state(state)
    visualizer.join()
