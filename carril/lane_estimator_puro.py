# -*- coding: utf-8 -*-
"""Estimador de carril en PYTHON PURO (sin numpy, sin OpenCV): misma logica que lane_estimator.py.

Existe porque el codigo a bordo de PenaltyStack es Python 2.7 puro y que haya numpy en el NAO esta por
verificar (docs/integracion_penaltystack.md, seccion 3). Si el NAO tiene numpy, se usa lane_estimator.py
(~10 veces mas rapido); si no, este.

Entrada: los bytes de la imagen tal como los entrega ALVideoDevice.getImageRemote (QVGA, RGB o BGR
intercalado) y la pose camara->robot de ALMotion.getTransform("CameraBottom", FRAME_ROBOT, True)
(lista de 16 floats). Salida: un objeto con valid, d, psi, n_lines, red_dist, quality.

Compatible con Python 2.7 y 3.
"""
from __future__ import division

import math
import random

HFOV_V6_DEG = 56.3
VFOV_V6_DEG = 43.7


class Estimacion(object):
    __slots__ = ("valid", "d", "psi", "n_lines", "c_left", "c_right", "slope", "n_points", "red_dist", "quality")

    def __init__(self):
        self.valid = False
        self.d = 0.0
        self.psi = 0.0
        self.n_lines = 0
        self.c_left = None
        self.c_right = None
        self.slope = 0.0
        self.n_points = 0
        self.red_dist = None
        self.quality = 0.0


class EstimadorCarrilPuro(object):
    def __init__(self, width=320, height=240, hfov_deg=HFOV_V6_DEG, vfov_deg=VFOV_V6_DEG,
                 lane_width=0.70, line_width=0.019, row_step=4, dark_ratio=0.62, min_contrast=10.0,
                 rel_contrast=0.18, local_window=41, width_tol=(0.35, 2.6), x_range=(0.12, 2.2), max_abs_y=1.6,
                 ransac_iters=60, inlier_tol=0.02, min_inliers=8, max_lines=4, pair_tol=0.08,
                 red_ratio=2.0, red_min=80, red_min_sat=70, red_min_row_px=25, red_min_rows=3,
                 color_order="rgb", polaridad="oscura", seed=0):
        self.w, self.h = int(width), int(height)
        self.fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
        self.cx, self.cy = (width - 1) / 2.0, (height - 1) / 2.0
        self.lane_width, self.line_width = lane_width, line_width
        self.spacing = lane_width + line_width
        self.rows = list(range(0, self.h, row_step))
        self.dark_ratio, self.min_contrast, self.rel_contrast = dark_ratio, min_contrast, rel_contrast
        self.clara = polaridad == "clara"   # "oscura" (cinta negra) o "clara" (cinta blanca sobre piso oscuro)
        self.k = int(local_window) | 1
        self.width_tol, self.x_range, self.max_abs_y = width_tol, x_range, max_abs_y
        self.ransac_iters, self.inlier_tol, self.min_inliers = ransac_iters, inlier_tol, min_inliers
        self.max_lines, self.pair_tol = max_lines, pair_tol
        self.red_ratio, self.red_min, self.red_min_sat = red_ratio, red_min, red_min_sat
        self.red_min_row_px, self.red_min_rows = red_min_row_px, red_min_rows
        self.ir, self.ib = (0, 2) if color_order == "rgb" else (2, 0)
        self.rng = random.Random(seed)
        self.last_points = []

    # ------------------------------------------------------------------ geometria

    def _project(self, u, v, T):
        """Pixel -> punto en el suelo (x, y) y profundidad, o None. T: lista de 16 floats (fila mayor)."""
        rx, ry, rz = 1.0, -(u - self.cx) / self.fx, -(v - self.cy) / self.fy
        dx = T[0] * rx + T[1] * ry + T[2] * rz
        dy = T[4] * rx + T[5] * ry + T[6] * rz
        dz = T[8] * rx + T[9] * ry + T[10] * rz
        if dz > -1e-6:
            return None
        s = -T[11] / dz
        return T[3] + s * dx, T[7] + s * dy, s

    # ------------------------------------------------------------------ deteccion

    def _scan(self, data, T):
        w, k, half = self.w, self.k, self.k // 2
        ir, ib = self.ir, self.ib
        lo_tol, hi_tol = self.width_tol
        pts = []
        red_rows = []
        for v in self.rows:
            base = v * w * 3
            row = data[base:base + w * 3]
            gray = [0.299 * row[3 * i + ir] + 0.587 * row[3 * i + 1] + 0.114 * row[3 * i + ib] for i in range(w)]
            # meta roja: contar pixeles rojos saturados en la fila
            nred, sum_u = 0, 0
            for i in range(0, w, 2):
                r, g, b = row[3 * i + ir], row[3 * i + 1], row[3 * i + ib]
                if r > self.red_min and r > self.red_ratio * g and r > self.red_ratio * b and r - max(g, b) > self.red_min_sat:
                    nred += 1
                    sum_u += i
            if 2 * nred >= self.red_min_row_px:
                red_rows.append((v, sum_u / float(nred)))
            # fondo local: media movil con borde replicado
            acc = gray[0] * (half + 1) + sum(gray[1:half + 1])
            run_start = -1
            for i in range(w + 1):
                if i < w:
                    j_in = i + half if i + half < w else w - 1
                    j_out = i - half - 1 if i - half - 1 > 0 else 0
                    if i > 0:
                        acc += gray[j_in] - gray[j_out]
                    bg = acc / k
                    gi = gray[i]
                    thr = self.min_contrast if self.min_contrast > self.rel_contrast * bg else self.rel_contrast * bg
                    if self.clara:
                        dark = gi > bg + (255.0 - bg) * (1.0 - self.dark_ratio) and (gi - bg) > thr
                    else:
                        dark = gi < bg * self.dark_ratio and (bg - gi) > thr
                else:
                    dark = False
                if dark and run_start < 0:
                    run_start = i
                elif not dark and run_start >= 0:
                    width = i - run_start
                    uc = (run_start + i - 1) / 2.0
                    run_start = -1
                    p = self._project(uc, float(v), T)
                    if p is None:
                        continue
                    x, y, depth = p
                    expected = self.fx * self.line_width / max(depth, 0.05)
                    if (width >= max(1.0, lo_tol * expected) and width <= hi_tol * expected + 2.0
                            and self.x_range[0] < x < self.x_range[1] and abs(y) < self.max_abs_y):
                        pts.append((x, y))
        red_dist = None
        if len(red_rows) >= self.red_min_rows:
            xs = []
            for v, uc in red_rows:
                p = self._project(uc, float(v), T)
                if p is not None:
                    xs.append(p[0])
            if len(xs) >= self.red_min_rows:
                xs.sort()
                red_dist = xs[int(0.1 * (len(xs) - 1))]
        return pts, red_dist

    # ------------------------------------------------------------------ rectas

    @staticmethod
    def _fit(pts):
        n = float(len(pts))
        xm = sum(p[0] for p in pts) / n
        ym = sum(p[1] for p in pts) / n
        vx = sum((p[0] - xm) ** 2 for p in pts)
        if vx < 1e-9:
            return None
        m = sum((p[0] - xm) * (p[1] - ym) for p in pts) / vx
        return ym - m * xm, m

    def _ransac_lines(self, pts):
        lines = []
        remaining = list(pts)
        tol = self.inlier_tol
        for _ in range(self.max_lines):
            n = len(remaining)
            if n < self.min_inliers:
                break
            best, best_n = None, 0
            for _it in range(self.ransac_iters):
                x1, y1 = remaining[self.rng.randrange(n)]
                x2, y2 = remaining[self.rng.randrange(n)]
                if abs(x2 - x1) < 0.08:
                    continue
                m = (y2 - y1) / (x2 - x1)
                if abs(m) > 1.2:
                    continue
                c = y1 - m * x1
                lim = tol * math.sqrt(1 + m * m)
                cnt = 0
                for (x, y) in remaining:
                    if abs(y - c - m * x) < lim:
                        cnt += 1
                if cnt > best_n:
                    best, best_n = (c, m), cnt
            if best is None or best_n < self.min_inliers:
                break
            c, m = best
            lim = tol * math.sqrt(1 + m * m)
            inl = [p for p in remaining if abs(p[1] - c - m * p[0]) < lim]
            if max(p[0] for p in inl) - min(p[0] for p in inl) > 0.15:
                f = self._fit(inl)
                if f is not None:
                    c, m = f
            lim = tol * math.sqrt(1 + m * m)
            inl_set = [abs(p[1] - c - m * p[0]) < lim for p in remaining]
            n_in = sum(inl_set)
            if n_in < self.min_inliers:
                break
            lines.append((c, m, n_in))
            remaining = [p for p, keep in zip(remaining, inl_set) if not keep]
        return lines

    def _choose(self, lines):
        best = None
        for a in range(len(lines)):
            for b in range(len(lines)):
                if a == b:
                    continue
                cl, ml, nl = lines[a]
                cr, mr, nr = lines[b]
                if not (cl > 0.0 > cr) or abs(ml - mr) > 0.15:
                    continue
                m = (ml * nl + mr * nr) / float(nl + nr)
                err = abs((cl - cr) / math.sqrt(1 + m * m) - self.spacing)
                if err > self.pair_tol:
                    continue
                score = nl + nr - 200.0 * err
                if best is None or score > best[0]:
                    best = (score, cl, cr, m)
        if best is not None:
            return best[1], best[2], best[3], 2
        cands = [l for l in lines if abs(l[0]) < self.spacing * 1.1]
        if not cands:
            return None, None, 0.0, 0
        c, m, n = max(cands, key=lambda l: (l[2], -abs(l[0])))
        off = self.spacing * math.sqrt(1 + m * m)
        if c > 0:
            return c, c - off, m, 1
        return c + off, c, m, 1

    # ------------------------------------------------------------------ API

    def procesar(self, data, T):
        """data: bytes/str/bytearray de la imagen (w*h*3). T: 16 floats camara->robot (fila mayor)."""
        if not isinstance(data, bytearray):
            data = bytearray(data)
        T = [float(t) for t in (T if len(T) == 16 else [v for fila in T for v in fila])]
        est = Estimacion()
        pts, est.red_dist = self._scan(data, T)
        self.last_points = pts
        est.n_points = len(pts)
        lines = self._ransac_lines(pts) if len(pts) >= self.min_inliers else []
        cl, cr, m, n_lines = self._choose(lines)
        if n_lines == 0:
            return est
        est.valid = True
        est.n_lines = n_lines
        est.c_left, est.c_right, est.slope = cl, cr, m
        est.d = -0.5 * (cl + cr) / math.sqrt(1 + m * m)
        est.psi = -math.atan(m)
        est.quality = 1.0 if n_lines == 2 else 0.5
        return est
