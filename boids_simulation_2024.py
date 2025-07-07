"""
Симуляция поведения "боидов" с неподвижными круглыми препятствиями
Реализовано: 

В соответствии с Вариантом -
3. добавить препятствия:
*неподвижные, круглой формы, радиусы различные, отталкивают агентов,
*визуализировать препятствия окружностями

Дополнительно -
4.внести изменения в функции расчета взаимодействий таким образом, чтобы:
*форма области, в которой агент видит соседей, была сектором заданного угла (90-270 градусов),
*ось сектора ориентирована вдоль вектора скорости,
*для одного агента визуализировать область видимости и попавших в нее агентов другим цветом

Студент: Серебров Борис
Группа: БПМ221
"""

import numpy as np
import math
from numba import njit, prange

from vispy import app, scene
from vispy.scene import Arrow, Text
from vispy.scene.visuals import Ellipse, Line, Markers

app.use_app('pyglet')
import time

import subprocess

# ======Съемка видео======

# Параметры записи
ffmpeg_path = r"C:\Users\mrche\Desktop\КП3\CP_HW2_2\ffmpeg\bin\ffmpeg.exe"
window_title = "Vispy Canvas"      # заголовок окна
fps = 60
duration   = "00:01:00"            # минута записи
output     = "boids_1000_60s.mp4"

# Собираем команду
cmd = [
    ffmpeg_path,
    "-y",                       # перезаписать без спроса
    "-f", "gdigrab",            # интерфейс захвата экрана на Windows
    "-framerate", str(fps),     # частота
    "-i", f"title={window_title}",  # именно это окно
    "-t", duration,             # длительность
    "-c:v", "libx264",          # кодек
    "-pix_fmt", "yuv420p",      # совместимый формат пикселей
    output
]

# ====== Конфигурация модели ======
WIDTH, HEIGHT = 1920, 1080       # размер окна (пиксели)
FPS = 60                          # частота кадров
DT = 1.0 / FPS                    # шаг по времени
N = 1000                          # число агентов
ASPECT = WIDTH / HEIGHT           # соотношение сторон области
PERCEPTION = 1.0 /4                 # радиус видимости
VIEW_ANGLE = math.pi / 2          # угол обзора (90°)
WALL_MARGIN = 0.05                # зона отталкивания от стен
VRANGE = (0.0, 0.1)               # диапазон скоростей
ARANGE = (0.0, 0.05)              # диапазон ускорений

# Параметры взаимодействий
# C – сплочение, A – выравнивание, S – разделение, W – отталкивание от стен
C, A, S, W = 0.09, 0.08, 0.18, 0.05

# Параметр шума: масштаб случайных неучтенных факторов
NOISE_SCALE = 0.01

# Препятствия: (x_center, y_center, radius)
OBSTACLES = np.array([
    (0.3, 0.3, 0.05),
    (0.7, 0.6, 0.08),
    (0.5, 0.5, 0.06),
], dtype=np.float64)

# Индекс агента для визуализации сектора
FOCUS_AGENT = 4


def initial_distribution(boids, rng):
    """
    Инициализирует случайные позиции и скорости агентов.

    boids: массив формы (N,6), колонки:
        0-1: координаты x,y
        2-3: компоненты скорости vx,vy
        4-5: ускорения (обнуляются)
    rng: генератор случайных чис
    """
    n = boids.shape[0]
    boids[:, 0] = rng.uniform(0, ASPECT, n)
    boids[:, 1] = rng.uniform(0, 1, n)
    ang = rng.uniform(0, 2 * math.pi, n)
    spd = rng.uniform(VRANGE[0], VRANGE[1], n)
    boids[:, 2] = spd * np.cos(ang)
    boids[:, 3] = spd * np.sin(ang)
    boids[:, 4:6] = 0.0


def vclip(v, vmin, vmax):
    """
    Обрезает векторы по модулю:
    - векторы с нормой > vmax уменьшаются до vmax.
    - ненулевые векторы с нормой < vmin увеличиваются до vmin.

    Аргументы:
        v: массив shape (N,2), векторы по строкам.
        vmin: минимальная допустимая норма.
        vmax: максимальная допустимая норма.
    """
    norms = np.linalg.norm(v, axis=1)
    mask = norms > vmax
    if np.any(mask):
        v[mask] *= (vmax / norms[mask]).reshape(-1, 1)
    if vmin > 0:
        mask = (norms > 0) & (norms < vmin)
        if np.any(mask):
            v[mask] *= (vmin / norms[mask]).reshape(-1, 1)


def propagate(boids):
    """
    Обновляет позиции и скорости агентов за один шаг времени,
    учитывает отражения от границ и препятствий.

    boids[:,2:4] обновляются по ускорениям boids[:,4:6],
    позиции boids[:,0:2] по скоростям.
    """
    vclip(boids[:, 4:6], ARANGE[0], ARANGE[1])
    boids[:, 2:4] += DT * boids[:, 4:6]
    vclip(boids[:, 2:4], VRANGE[0], VRANGE[1])
    boids[:, 0:2] += DT * boids[:, 2:4]
    # отскок от границ
    for d, low, high in ((0, 0, ASPECT), (1, 0, 1)):
        mask_low = boids[:, d] < low
        boids[mask_low, d] = low
        boids[mask_low, d + 2] *= -1
        mask_high = boids[:, d] > high
        boids[mask_high, d] = high
        boids[mask_high, d + 2] *= -1
    # отскок от препятствий
    for xo, yo, r in OBSTACLES:
        dx = boids[:, 0] - xo
        dy = boids[:, 1] - yo
        dist = np.hypot(dx, dy)
        idx = np.where(dist < r)[0]
        for i in idx:
            nx, ny = dx[i] / dist[i], dy[i] / dist[i]
            boids[i, 0] = xo + nx * r
            boids[i, 1] = yo + ny * r
            vdot = boids[i, 2] * nx + boids[i, 3] * ny
            boids[i, 2] -= 2 * vdot * nx
            boids[i, 3] -= 2 * vdot * ny


def directions(boids):
    """
    Возвращает массив точек для отрисовки стрелок:
    попарно (start, end) для каждого агента.
    """
    return np.hstack((boids[:, :2] - DT * boids[:, 2:4], boids[:, :2]))

@njit(cache=True, parallel=True)
def compute_accelerations(boids, noise):
    """
    Вычисляет ускорения агентов по правилам поведения:
    - C: сплочение (cohesion)
    - A: выравнивание (alignment)
    - S: разделение (separation)
    - W: отталкивание от стен (wall avoidance)
    - boids: numpy.ndarray, форма (N,6)
    - noise: numpy.ndarray, форма (N,2) добавочная шумовая компонента
    Перезаписывает boids[:,4:6] новым ускорением.
    """
    n = boids.shape[0]
    p2 = PERCEPTION**2
    for i in prange(n):
        x, y, vx, vy = boids[i, 0], boids[i, 1], boids[i, 2], boids[i, 3]
        cnt = sx = sy = svx = svy = rx = ry = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = boids[j, 0] - x
            dy = boids[j, 1] - y
            d2 = dx * dx + dy * dy
            if d2 < p2:
                d = math.sqrt(d2)
                sx += boids[j, 0]
                sy += boids[j, 1]
                svx += boids[j, 2]
                svy += boids[j, 3]
                cnt += 1
                if d > 1e-8:
                    rx += -dx / d
                    ry += -dy / d
        coh_x = coh_y = alg_x = alg_y = sep_x = sep_y = 0.0
        if cnt > 0:
            cx, cy = sx / cnt, sy / cnt
            mvx, mvy = svx / cnt, svy / cnt
            coh_x = (cx - x) / PERCEPTION
            coh_y = (cy - y) / PERCEPTION
            alg_x = (mvx - vx) / (2 * VRANGE[1])
            alg_y = (mvy - vy) / (2 * VRANGE[1])
            rx /= cnt
            ry /= cnt
            nr = math.hypot(rx, ry)
            if nr > 1e-8:
                rx /= nr
                ry /= nr
            sep_x, sep_y = rx, ry
        wx = wy = 0.0
        tx = max(0.0, min(1.0, (x - ASPECT * WALL_MARGIN) / (-ASPECT * WALL_MARGIN)))
        wx += tx * tx * (3 - 2 * tx)
        tx = max(0.0, min(1.0, (x - ASPECT * (1 - WALL_MARGIN)) / (ASPECT * WALL_MARGIN)))
        wx -= tx * tx * (3 - 2 * tx)
        ty = max(0.0, min(1.0, (y - WALL_MARGIN) / (-WALL_MARGIN)))
        wy += ty * ty * (3 - 2 * ty)
        ty = max(0.0, min(1.0, (y - (1 - WALL_MARGIN)) / WALL_MARGIN))
        wy -= ty * ty * (3 - 2 * ty)
        boids[i, 4] = C * coh_x + A * alg_x + S * sep_x + W * wx + noise[i, 0]
        boids[i, 5] = C * coh_y + A * alg_y + S * sep_y + W * wy + noise[i, 1]

if __name__ == '__main__':
    boids = np.zeros((N, 6), dtype=np.float64)
    rng = np.random.default_rng(1)
    initial_distribution(boids, rng)

    canvas = scene.SceneCanvas(show=True, size=(WIDTH, HEIGHT), bgcolor='black')
    view = canvas.central_widget.add_view()
    view.camera = scene.PanZoomCamera(rect=(0, 0, ASPECT, 1))

    # рисуем препятствия
    for xo, yo, r in OBSTACLES:
        Ellipse(center=(xo, yo), radius=r, border_color='red', color=None, parent=view.scene)

    arrows = Arrow(arrows=directions(boids), arrow_color=(0, 0.7, 1, 1), arrow_size=8, connect='segments', parent=view.scene)

    # сектор видимости и подсветка соседей
    vis_sector = Line(color=(0, 1, 0, 0.3), connect='strip', parent=view.scene)
    vis_neighbors = Markers(symbol='disc', size=6, face_color=(0, 1, 0, 1), parent=view.scene)

    # HUD
    info1 = Text('', color='white', parent=view.scene)
    info2 = Text('', color='white', parent=view.scene)
    info1.font_size = 12
    info2.font_size = 12
    info1.pos = (0.02 * ASPECT + 0.2, 0.98)
    info2.pos = (0.02 * ASPECT + 0.2, 0.94)

    last = time.perf_counter()
    count = 0
    fps = 0.0


    def update(ev):
        global last, count, fps
        # генерация шума масштабированного неучтёнными факторами
        noise = rng.uniform(-1, 1, (N, 2)) * NOISE_SCALE
        compute_accelerations(boids, noise)
        propagate(boids)
        arrows.set_data(arrows=directions(boids))

        # сектор обзора для focus-agent
        x0, y0 = boids[FOCUS_AGENT, 0], boids[FOCUS_AGENT, 1]
        vx, vy = boids[FOCUS_AGENT, 2], boids[FOCUS_AGENT, 3]
        angle = math.atan2(vy, vx)
        half = VIEW_ANGLE / 2
        angles = np.linspace(angle - half, angle + half, 60)
        poly = np.column_stack((x0 + np.cos(angles) * PERCEPTION, y0 + np.sin(angles) * PERCEPTION))
        polygon = np.vstack(((x0, y0), poly, (x0, y0)))
        vis_sector.set_data(polygon)

        visible = []
        for j in range(N):
            if j == FOCUS_AGENT:
                continue
            dx, dy = boids[j, 0] - x0, boids[j, 1] - y0
            d2 = dx * dx + dy * dy
            if d2 <= PERCEPTION * PERCEPTION:
                ang2 = math.atan2(dy, dx)
                da = ((ang2 - angle + math.pi) % (2 * math.pi)) - math.pi
                if abs(da) <= half:
                    visible.append((boids[j, 0], boids[j, 1]))
        if visible:
            vis_neighbors.set_data(np.array(visible), face_color=(0,1,0,1), size=6)
        else:
            vis_neighbors.set_data(np.zeros((0,2)), face_color=(0,1,0,1), size=6)

        count += 1
        now = time.perf_counter()
        if now - last >= 1.0:
            fps = count / (now - last)
            count = 0
            last = now
        info1.text = f"N={N}, c={C:.2f}, a={A:.2f}, s={S:.2f}, w={W:.2f}, noise={NOISE_SCALE:.2f}"
        info2.text = f"PER={PERCEPTION:.2f}, FPS={fps:.1f}"
        canvas.update()
    timer = app.Timer(interval=DT, connect=update, start=True)
    subprocess.Popen(cmd,
                 stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL)
    app.run()
