import sys
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

UNIT_meter = 1/5
DISSIPATION = 1.0
BG_COLOR = "#fff"
STICK_COLOR = "#f00"
WALL_COLOR = "#000"
CLOSE = 1E-3
T_RESOLUTION = 67
SPEED = 0.01

Dt = 0.001

# Stick
# np.array([x (0), y (1), phi (2), vx (3), vy (4), w (5), R (6), m (7)])

# FixedStick
# np.array([x (0), y (1), phi (2), R (3)])


def Energ(stick):
    I_nd = 1/3
    return stick[3]**2 + stick[4]**2 + stick[5]**2*I_nd*stick[6]**2


def _1d_endpoints(params):
    # params: x, y, phi, R
    shift_x, shift_y = params[3]*np.cos(params[2]), params[3]*np.sin(params[2])
    x1, y1 = params[0] - shift_x, params[1] - shift_y
    x2, y2 = params[0] + shift_x, params[1] + shift_y
    return np.array([x1, y1, x2, y2])


def _2d_endpoints(params):
    # params: x, y, phi, R
    shift_x, shift_y = params[:, 3]*np.cos(params[:, 2]), params[:, 3]*np.sin(params[:, 2])
    x1, y1 = params[:, 0] - shift_x, params[:, 1] - shift_y
    x2, y2 = params[:, 0] + shift_x, params[:, 1] + shift_y
    return np.vstack([x1, y1, x2, y2]).T


def _update_lines(lines, data):
    for line, coords in zip(lines, data):
        px_coords = coords/UNIT_meter
        line.set_xdata([px_coords[0], px_coords[2]])
        line.set_ydata([px_coords[1], px_coords[3]])


def update_sticks(lines, sticks):
    sticks_endpoints = _2d_endpoints(sticks[:, [0, 1, 2, 6]])
    _update_lines(lines, sticks_endpoints)


def update_walls(lines, walls):
    _update_lines(lines, _2d_endpoints(walls))


def evolve_sticks(sticks):
    new_sticks = np.copy(sticks)
    new_sticks[:, 0] = sticks[:, 0] + sticks[:, 3]*Dt
    new_sticks[:, 1] = sticks[:, 1] + sticks[:, 4]*Dt
    new_sticks[:, 2] = (sticks[:, 2] + sticks[:, 5]*Dt) % (2*np.pi)
    return new_sticks


def is_collision(stick, wall):
    x1, y1, x2, y2 = _1d_endpoints(stick[[0, 1, 2, 6]])
    y_test = (x1 - wall[0])*np.tan(wall[2]) + wall[1]
    if (y_test - y1)**2 < CLOSE:
        return 1
    y_test = (x2 - wall[0])*np.tan(wall[2]) + wall[1]
    if (y_test - y2)**2 < CLOSE:
        return 2
    return 0


def collide_stick_wall(stick, wall, stick_endpoint):
    x1, y1, x2, y2 = _1d_endpoints(stick[[0, 1, 2, 6]])
    x, y = (x1, y1) if stick_endpoint == 1 else (x2, y2)
    phi = (stick[2] - wall[2]) % np.pi
    I_nd = 1/3
    vcy0 = stick[4]*np.cos(wall[2]) - stick[3]*np.sin(wall[2])
    dv = DISSIPATION*(stick[5]*stick[6]*np.cos(phi) - vcy0)/(0.5 + np.cos(phi)**2/I_nd*0.5)
    new_stick = np.copy(stick)
    new_stick[0] += np.sqrt(10*CLOSE)*(-np.sin(wall[2]))
    new_stick[1] += np.sqrt(10*CLOSE)*np.cos(wall[2])
    new_stick[3] += dv*(-np.sin(wall[2]))
    new_stick[4] += dv*np.cos(wall[2])
    new_stick[5] -= dv*np.cos(phi)/I_nd/stick[6]
    return new_stick


def collide_sticks_walls(sticks, walls):
    new_sticks = np.copy(sticks)
    changed = False
    for i, stick in enumerate(sticks):
        print(Energ(stick))
        for wall in walls:
            stick_eid = is_collision(stick, wall)
            if not stick_eid:
                continue
            changed = 2 if stick_eid == 2 else False
            new_sticks[i] = collide_stick_wall(stick, wall, stick_eid)
    # if changed:
        # raise Exception(new_sticks)
    return new_sticks


def main():
    # sticks = np.array([[12., 9., np.pi/3, 0.2, -4, 1.2, 1, 3.],
                       # [16., 10., np.pi+np.pi/3, 0.2, -4, -1.2, 1, 3.]])
    sticks = np.array([[12., 9., np.pi/2, 0.2, -4, 0., 1, 3.]])
    stick_lines = [plt.plot([], [], color=STICK_COLOR)[0] for i in range(sticks.shape[0])]
    walls = np.array([[3, 7., 0*np.pi/14, 50.],
                      [3, 14., np.pi-np.pi/14, 50.]])
    wall_lines = [plt.plot([], [], color=WALL_COLOR)[0] for i in range(walls.shape[0])]

    plt.axis([0, 100, 0, 100])
    plt.xlim([0, 100])
    plt.ylim([0, 100])

    update_walls(wall_lines, walls)
    plt.show()
    plt.pause(SPEED)

    cnt = 0

    while True:
        new_sticks = evolve_sticks(sticks)
        update_sticks(stick_lines, new_sticks)
        try:
            new_sticks = collide_sticks_walls(new_sticks, walls)
        except Exception as e:
            new_sticks = e.args[0]
            # T_RESOLUTION = 1
            input()
        sticks = new_sticks
        print(sticks)
        if (cnt % T_RESOLUTION) == 0:
            plt.pause(SPEED)
            if T_RESOLUTION == 1:
                input()
            cnt = 0
        cnt += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
