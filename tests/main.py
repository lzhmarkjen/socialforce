from contextlib import contextmanager
import numpy as np
import pytest
import socialforce


# 穿过门
@pytest.mark.plot
def test_gate():
    door_x = 0.0  # 门的横坐标
    door_width = 1.4  # 门宽
    n = 10  # 人数
    destination = [5.0, 0.0]  # 目的地坐标
    range_x = [-10, door_x]  # 初始状态人群位置的x范围
    range_y = [-6.0, 6.0]  # 初始状态人群位置的y范围
    vel_x = 1.0  # x方向速度
    vel_y = 1.0  # y方向速度

    x_pos = np.random.random((n, 1)) * np.array([range_x[0]])
    y_pos = ((np.random.random((n, 1)) - 0.5) * 2.0) * np.array(([range_y[1]]))
    x_vel = np.full((n, 1), vel_x)
    y_vel = np.full((n, 1), vel_y)
    x_dest = np.full((n, 1), destination[0])
    y_dest = np.full((n, 1), destination[1])
    initial_state = np.concatenate((x_pos, y_pos, x_vel, y_vel, x_dest, y_dest), axis=-1)
    print(initial_state)

    space = [
        np.array([(door_x, y) for y in np.linspace(-10, -door_width / 2, 1000)]),
        np.array([(door_x, y) for y in np.linspace(door_width / 2, 10, 1000)]),
    ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space))

    # 判断是否所有人都通过了门
    states = []
    i = 0
    while True:
        i += 1
        state = s.step().state.copy()
        states.append(state)
        end = True
        for sta in state:
            if sta[0] < 0:
                end = False
        if end:
            # print("terminate when time is ", i)
            break

    states = np.stack(states)

    # 生成动态图
    with visualize(states, space, 'out/gate.gif') as _:
        pass
    # 生成路线图
    with socialforce.show.canvas('out/gate.png') as ax:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        for ped in range(len(initial_state)):
            x = states[:, ped, 0]
            y = states[:, ped, 1]
            ax.plot(x, y, '-o', label='ped {}'.format(ped), markersize=2.5)
        ax.set_title("evacuate time {}".format(i))
        ax.legend()


# 人行道对穿
@pytest.mark.plot
def test_walkway():
    width = 5.0  # 人行道宽度
    n = 100  # 行人数量
    length = 25.0  # 人行道长度
    vel = [1.34, 0.26]  # 速度服从高斯分布

    pos_left = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([length, width])
    pos_right = ((np.random.random((n, 2)) - 0.5) * 2.0) * np.array([length, width])

    x_vel_left = np.random.normal(vel[0], vel[1], size=(n, 1))
    x_vel_right = np.random.normal(-vel[0], vel[1], size=(n, 1))
    x_destination_left = 100.0 * np.ones((n, 1))
    x_destination_right = -100.0 * np.ones((n, 1))
    zeros = np.zeros((n, 1))
    state_left = np.concatenate((pos_left, x_vel_left, zeros, x_destination_left, zeros), axis=-1)
    state_right = np.concatenate((pos_right, x_vel_right, zeros, x_destination_right, zeros), axis=-1)
    initial_state = np.concatenate((state_left, state_right))

    space = [
        np.array([(x, width) for x in np.linspace(-length, length, num=5000)]),
        np.array([(x, -width) for x in np.linspace(-length, length, num=5000)]),
    ]
    s = socialforce.Simulator(initial_state, socialforce.PedSpacePotential(space))
    states = []
    densities = []
    i = 0
    while True:
        density = 0
        i += 1
        state = s.step().state
        states.append(state.copy())

        for sta in state:
            if -length < sta[0] < length:
                density += 1
        densities.append([i, density])
        if density == 0:
            break

    states = np.stack(states)
    densities = np.stack(densities)

    with visualize(states, space, 'out/walkway_{}.gif'.format(n)) as _:
        pass
    with socialforce.show.canvas('out/walkway_{}.png'.format(n)) as ax:
        ax.set_xlabel('t')
        ax.set_ylabel('density')
        x = densities[:, 0]
        y = densities[:, 1]
        ax.plot(x, y, '-o', label='density', markersize=2.5)
        ax.legend()


@contextmanager
def visualize(states, space, output_filename):
    import matplotlib.pyplot as plt

    print('')
    with socialforce.show.animation(
            len(states),
            output_filename,
            writer='imagemagick') as context:
        ax = context['ax']
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        yield ax

        for s in space:
            ax.plot(s[:, 0], s[:, 1], '-o', color='black', markersize=2.5)

        actors = []
        for ped in range(states.shape[1]):
            speed = np.linalg.norm(states[0, ped, 2:4])
            radius = 0.2 + speed / 2.0 * 0.3
            p = plt.Circle(states[0, ped, 0:2], radius=radius,
                           facecolor='black' if states[0, ped, 4] > 0 else 'white',
                           edgecolor='black')
            actors.append(p)
            ax.add_patch(p)

        def update(i):
            for ped, p in enumerate(actors):
                # p.set_data(states[i:i+5, ped, 0], states[i:i+5, ped, 1])
                p.center = states[i, ped, 0:2]
                speed = np.linalg.norm(states[i, ped, 2:4])
                p.set_radius(0.2 + speed / 2.0 * 0.3)

        context['update_function'] = update
