import numpy as np
def _world_to_pixel(self, x, y):
    px = int((x - self.xmin) / self.bev_res)
    py = int((self.ymax - y) / self.bev_res)
    return px, py

def _box_corners_bev(self, cx, cy, dx, dy, yaw):

    half_dx, half_dy = dx/2, dy/2

    corners = np.array([
        [ half_dx,  half_dy],
        [ half_dx, -half_dy],
        [-half_dx, -half_dy],
        [-half_dx,  half_dy],
    ])

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])

    corners = corners @ rot.T
    corners[:, 0] += cx
    corners[:, 1] += cy

    return corners