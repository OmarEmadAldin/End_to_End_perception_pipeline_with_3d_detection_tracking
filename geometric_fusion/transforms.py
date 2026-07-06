import numpy as np
# There's ros package that do that (the code from the internet)
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

# Sensor frame  to Global frame
def sensor_to_global(p_sensor, calib, ego):
    
    R_s2e = quat_to_rot(calib['rotation'])
    t_s2e = np.array(calib['translation'])
    R_e2g = quat_to_rot(ego['rotation'])
    t_e2g = np.array(ego['translation'])

    p_ego    = R_s2e @ p_sensor + t_s2e
    p_global = R_e2g @ p_ego    + t_e2g
    return p_global

#Velocity: sensor frame  →  global frame
def vel_sensor_to_global(vx, vy, calib, ego):
   
    R_s2e = quat_to_rot(calib['rotation'])
    R_e2g = quat_to_rot(ego['rotation'])

    v_sensor = np.array([vx, vy, 0.0])
    v_ego    = R_s2e @ v_sensor
    v_global = R_e2g @ v_ego
    return v_global


def project_to_image(p_cam, K):
  
    X, Y, Z = p_cam
    if Z <= 0.1:
        return None                         

    uvw = K @ np.array([X, Y, Z])          # homogeneous pixel coords
    return np.array([uvw[0] / uvw[2],      # u = fx·(X/Z) + cx
                     uvw[1] / uvw[2]])      # v = fy·(Y/Z) + cy
