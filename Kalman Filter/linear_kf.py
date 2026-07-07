import numpy as np

C_LIDAR = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)
R_LIDAR = np.diag([0.5**2, 0.5**2]) 

C_RADAR = np.eye(4, dtype=float)
R_RADAR = np.diag([
    1.0**2,   # position x uncertainty 
    1.0**2,   # position y uncertainty 
    0.3**2,   # velocity x uncertainty 
    0.3**2,   # velocity y uncertainty 
])

class LinearKalmanFilter:
    def __init__(self, init_x, init_y, init_vx=0.0, init_vy=0.0, q_var=1.0):
        self.q_var = q_var
        self.x = np.array([init_x, init_y, init_vx, init_vy], dtype=float)
        self.P = np.diag([
            0.5**2,   # px uncertainty (m²)
            0.5**2,   # py uncertainty (m²)
            5.0**2,   # vx uncertainty — large because we don't know it yet
            5.0**2,   # vy uncertainty
        ])

    def predict(self, dt):
        if dt <= 0:
            return

        # State transition matrix
        A = np.array([
            [1, 0, dt,  0],
            [0, 1,  0, dt],
            [0, 0,  1,  0],
            [0, 0,  0,  1],
        ], dtype=float)

        q = self.q_var
        Q = q * np.array([
            [dt**4 / 4,  0,          dt**3 / 2,  0         ],
            [0,          dt**4 / 4,  0,          dt**3 / 2 ],
            [dt**3 / 2,  0,          dt**2,      0         ],
            [0,          dt**3 / 2,  0,          dt**2     ],
        ])

        self.x = A @ self.x
        self.P = A @ self.P @ A.T + Q

    def update(self, z, C, R):
        """
        Equations:
            y  = z − H · x̂           output state (how wrong the prediction was)
            S  = C · P · Cᵀ + R      output covariance (uncertainty of y)
            K  = P · Hᵀ · S⁻¹        Kalman gain (how much to trust measurement)
            x̂  = x̂ + K · y           corrected state
            P  = (I − K · C) · P     corrected covariance (uncertainty reduced)
         """
        y  = z - C @ self.x
        S  = C @ self.P @ C.T + R
        K  = self.P @ C.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ C) @ self.P

    def update_lidar(self, px, py):
        self.update(np.array([px, py]), C_LIDAR, R_LIDAR)

    def update_radar(self, px, py, vx, vy):
        self.update(np.array([px, py, vx, vy]), C_RADAR, R_RADAR)

    # property is a getter attribute method
    @property
    def position(self):
        return self.x[:2].copy()

    @property
    def velocity(self):
        return self.x[2:].copy()

    @property
    def state(self):
        return self.x.copy()
