# Kalman Filter — Linear
### `kalman_filter/linear_kf.py`

---

## 1. What the Filter Does

The Kalman filter tracks **one object** over time. At every moment it maintains
two things:

- **`x`** — its best estimate of the object's state (position + velocity)
- **`P`** — its uncertainty about that estimate

It alternates between two steps forever:

```
┌──────────────────────────────────────────────────────────┐
│  PREDICT   (time passes, object moves)                   │
│     x̂ = A · x                                            │
│     P  = A · P · Aᵀ + Q                                  │
│                                                          │
│  UPDATE    (a sensor gives a new measurement)            │
│     y  = z  −  C · x̂           Output                    │
│     S  = H · P · Cᵀ + R        Output covariance         │
│     K  = P · Cᵀ · S⁻¹          Kalman gain               │
│     x̂  = x̂ + K · y             corrected state           │
│     P  = (I − K · C) · P       corrected covariance      │   
│  X -- > state (which is 4 states)                        │
│  P -- > Error Covariance Matrix                          │
│  Q -- > Process Noise                                    │
│  C -- > Mesurement Matrix                                │
│  R -- > Measurement Noise Covariance                     │
└──────────────────────────────────────────────────────────┘
```

---

## 2. The State Vector `x`  —  shape (4,1)

```
x = [px,  py,  vx,  vy]ᵀ

px = position  in x  (metres, global map frame)
py = position  in y  (metres, global map frame)
vx = velocity  in x  (m/s,   global map frame)
vy = velocity  in y  (m/s,   global map frame)
```

No sensor gives you all four at once:

```
LiDAR  →  gives px, py      (no velocity)
Radar  →  gives px, py, vx, vy
Camera →  gives pixels only  (no metric info)
```

The filter estimates all four continuously, even when only some are measured.

While X shape is (4,1) so A will be (4,4)

---

## 3. The Covariance Matrix `P`  —  shape (4×4)

`P` is the filter's **uncertainty** about every element of `x`, and how
those uncertainties are correlated with each other.

### Initial P Only diagonal

```python
P  = diag([0.25,  0.25,  25.0,  25.0])
```

**P shrinks after every update** (we become more certain). and the reamininng matrix parameters are updated not just the diagonal

---

## 4. PREDICT STEP

### 4.1 The F Matrix (State Transition)  —  shape (4×4)

F encodes the **physics of motion** between two timestamps separated by `dt` seconds,
under the constant-velocity assumption:

```
Physics equations:
    px_new = px + vx · dt       (position changes due to velocity)
    py_new = py + vy · dt
    vx_new = vx                 (velocity unchanged — constant velocity model)
    vy_new = vy

Written as a matrix multiplication  x_new = F · x_old:

A = | 1  0  dt   0  |     row 0: px_new = 1·px + 0·py + dt·vx + 0·vy
    | 0  1   0  dt  |     row 1: py_new = 0·px + 1·py + 0·vx + dt·vy
    | 0  0   1   0  |     row 2: vx_new = 0·px + 0·py + 1·vx + 0·vy
    | 0  0   0   1  |     row 3: vy_new = 0·px + 0·py + 0·vx + 1·vy
```

### 4.2 State Prediction

```
x̂ = F · x

Dimensions: (4×4) · (4,) = (4,)
```

Every element of the new state is a linear combination of the old state.
No sensor involved — pure physics.

### 4.3 The Q Matrix (Process Noise)  —  shape (4×4)

Objects don't move at **perfectly** constant velocity — they speed up, slow down,
turn. `Q` models this uncertainty. It is derived by integrating the effect of
a random acceleration `a(t)` with variance `q` over time `dt`:

### 4.4 Covariance Prediction

```
P = A · P · Aᵀ + Q

Dimensions: (4×4)·(4×4)·(4×4) + (4×4) = (4×4)
```


---

## 5. UPDATE STEP

The update step happens when a sensor delivers a new measurement `z`.
Different sensors give different `z`, `H`, and `R`.

### 5.1 The z Vector (Measurement)

```
LiDAR:   z = [px_measured,  py_measured]             shape (2,1)
Radar:   z = [px_measured,  py_measured,  vx,  vy]   shape (4,1)
```

### 5.2 The H Matrix (Measurement Model)

H answers: **"which elements of the state does this sensor observe?"**

It maps the state space (4D) into the measurement space (2D or 4D).

**H for LiDAR  —  shape (2×4):**

```
C_lidar = | 1  0  0  0 |
          | 0  1  0  0 |

Row 0: "I observe px"  →  [1,0,0,0] · [px,py,vx,vy] = px  
Row 1: "I observe py"  →  [0,1,0,0] · [px,py,vx,vy] = py  

```

**H for Radar  —  shape (4×4):**

```
C_radar = I₄ = | 1  0  0  0 |
               | 0  1  0  0 |
               | 0  0  1  0 |
               | 0  0  0  1 |

C · x = [px, py, vx, vy]   (radar observes the full state)
```

### 5.3 The R Matrix (Measurement Noise Covariance)

R encodes **how noisy the sensor is**. Larger R = noisier sensor = trust it less.

**R for LiDAR  —  shape (2×2):**

```
R_lidar = diag([0.25,  0.25])   metres²

```

**R for Radar  —  shape (4×4):**

```
R_radar = diag([1.0,   1.0,  0.09, 0.09])

```

### 5.4 The Full Cycle Equations

---

**Equations**

```
y = Z − C · x̂ (then multiplied by k for state update)

S = C · P · Cᵀ + R (then used in kalman gain)

K = P · Cᵀ · S⁻¹ 

x̂ = x̂ + K · y (state update)

P = (I − K · H) · P (Error update)


```
