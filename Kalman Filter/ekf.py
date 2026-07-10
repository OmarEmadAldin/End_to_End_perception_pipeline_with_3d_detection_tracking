import numpy as np
from kalman_filter.linear_kf import LinearKalmanFilter, C_LIDAR, R_LIDAR, C_RADAR, R_RADAR


R_RADAR_POLAR = np.diag([
    0.3**2,    # ρ  uncertainty (m²)     — range
    0.03**2,   # φ  uncertainty (rad²)   — bearing
    0.3**2,    # ρ̇  uncertainty (m²/s²)  — radial velocity
])


class ExtendedKalmanFilter(LinearKalmanFilter):
    def __init__(self, init_x, init_y, init_vx=0.0, init_vy=0.0, q_var=1.0):

        super().__init__(init_x, init_y, init_vx, init_vy, q_var=q_var)

    def update_radar_polar(self, rho, phi, rhodot):
        
        x_pred = self.x.copy()
        px, py, vx, vy = self.x
        rho_hat = np.sqrt(px**2 + py**2)
        if rho_hat < 1e-4:
            return          
        
        phi_hat    = np.arctan2(py, px)
        rhodot_hat = (px * vx + py * vy) / rho_hat
        h_x = np.array([rho_hat, phi_hat, rhodot_hat])

        z = np.array([rho, phi, rhodot])
        y = z - h_x
        y[1] = (y[1] + np.pi) % (2 * np.pi) - np.pi
        x_upd = self.x.copy()

        self.history["predicted"].append(x_pred)
        self.history["measurement"].append(x_upd)
        self.history["residual"].append(x_upd - x_pred)

        r2 = rho_hat**2    # ρ²
        r3 = rho_hat**3    # ρ³

        Hj = np.array([
            [px / rho_hat,py / rho_hat,0,0],
            [-py / r2,px / r2,0,0],
            [(vx * py**2 - vy * px * py) / r3,(vy * px**2 - vx * px * py) / r3,px / rho_hat,py / rho_hat],
        ])

        S = Hj @ self.P @ Hj.T + R_RADAR_POLAR
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ Hj) @ self.P

    def update_radar_cartesian(self, px, py, vx, vy):

        self.update(np.array([px, py, vx, vy]), C_RADAR, R_RADAR)

    def update_lidar(self, px, py):
        self.update(np.array([px, py]), C_LIDAR, R_LIDAR)

    def get_history(self):
        return self.history
    
    @staticmethod
    def cartesian_to_polar(px, py, vx, vy):
        rho    = np.sqrt(px**2 + py**2)
        phi    = np.arctan2(py, px)
        rhodot = (px * vx + py * vy) / max(rho, 1e-4)
        return rho, phi, rhodot