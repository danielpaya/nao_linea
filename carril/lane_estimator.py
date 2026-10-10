# -*- coding: utf-8 -*-
"""Estimacion de la posicion del NAO dentro del carril a partir de una imagen de la camara inferior.

Pipeline (todo en numpy, pensado para el Atom del NAO):
  1. Se toman filas espaciadas de la imagen y se pasan a escala de grises.
  2. En cada fila se buscan "tramos oscuros" (la cinta negra sobre la duela) cuyo ancho en pixeles sea
     compatible con una linea de 1.9 cm a esa distancia. Esto descarta la veta de la madera (muy delgada),
     la franja de salida y sombras grandes (muy anchas).
  3. El centro de cada tramo se proyecta al suelo con la pose de la camara (ALMotion.getTransform).
  4. RANSAC encuentra rectas en el suelo y se elige el par paralelo separado ~72 cm que encierra al robot.
     Si solo se ve una linea, se usa el ancho conocido del carril.
  5. Salida: desplazamiento lateral d (m, + = robot a la izquierda del centro) y error de rumbo psi
     (rad, + = robot girado a la izquierda respecto al carril). Ademas, distancia a la franja roja de meta.
"""
from __future__ import division

import math

import numpy as np


class LaneEstimate(object):
    __slots__ = ("valid", "d", "psi", "n_lines", "c_left", "c_right", "slope", "n_points",
                 "red_dist", "red_points", "quality")

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
        self.red_points = 0
        self.quality = 0.0

    def as_dict(self):
        return dict((k, getattr(self, k)) for k in self.__slots__)


class LaneEstimator(object):
    def __init__(self, camera,
                 lane_width=0.70,        # ancho interior del carril [m] (reglamento 3.2.4)
                 line_width=0.019,       # ancho de la linea negra [m]
                 row_step=3,             # se procesa 1 de cada row_step filas
                 row_start_frac=0.0,     # fraccion superior de la imagen a ignorar
                 dark_ratio=0.62,        # pixel oscuro si gris < dark_ratio * fondo de la fila
                 min_contrast=10.0,      # contraste minimo absoluto (0..255)
                 width_tol=(0.35, 2.6),  # ancho aceptado = [a, b] * ancho esperado (+2 px)
                 x_range=(0.12, 2.2),    # rango de distancia util en el suelo [m]
                 max_abs_y=1.6,
                 ransac_iters=70,
                 inlier_tol=0.02,        # [m]
                 min_inliers=8,
                 max_lines=4,
                 pair_tol=0.08,          # tolerancia en la separacion entre lineas [m]
                 red_ratio_g=2.0, red_ratio_b=2.0, red_min=80, red_min_sat=70, red_min_row_px=25,
                 red_min_points=40,
                 local_window=41,        # ancho [px] de la media movil para el fondo local (0 = mediana de la fila)
                 rel_contrast=0.18,      # contraste minimo relativo al fondo
                 refine=False,           # autocorreccion de pitch/roll (experimental: no mejoro en pruebas)
                 refine_pitch_deg=5.0, refine_roll_deg=4.0,
                 color_order="bgr",     # "bgr" (kBGRColorSpace) o "rgb" (kRGBColorSpace, el de PenaltyStack)
                 polaridad="oscura",    # "oscura": linea mas oscura que el piso (cinta negra sobre madera)
                                        # "clara":  linea mas clara que el piso (cinta blanca sobre piso oscuro)
                 seed=0):
        self.cam = camera
        self.lane_width = lane_width
        self.line_width = line_width
        self.spacing = lane_width + line_width  # distancia entre los centros de las dos lineas
        self.row_step = row_step
        self.row_start = int(row_start_frac * camera.height)
        self.dark_ratio = dark_ratio
        self.min_contrast = min_contrast
        self.width_tol = width_tol
        self.x_range = x_range
        self.max_abs_y = max_abs_y
        self.ransac_iters = ransac_iters
        self.inlier_tol = inlier_tol
        self.min_inliers = min_inliers
        self.max_lines = max_lines
        self.pair_tol = pair_tol
        self.red_ratio_g = red_ratio_g
        self.red_ratio_b = red_ratio_b
        self.red_min = red_min
        self.red_min_sat = red_min_sat
        self.red_min_row_px = red_min_row_px
        self.red_min_points = red_min_points
        self.local_window = int(local_window) | 1 if local_window else 0
        self.rel_contrast = rel_contrast
        if polaridad not in ("oscura", "clara"):
            raise ValueError('polaridad debe ser "oscura" o "clara"')
        self.polaridad = polaridad
        self.refine = refine
        self.refine_pitch = np.radians(np.arange(-refine_pitch_deg, refine_pitch_deg + 1e-6, 0.5))
        self.refine_roll = np.radians(np.arange(-refine_roll_deg, refine_roll_deg + 1e-6, 1.0))
        self._uv = np.zeros((0, 2))
        self.ib, self.ir = (0, 2) if color_order == "bgr" else (2, 0)
        self.rng = np.random.RandomState(seed)
        self.rows = np.arange(self.row_start, camera.height, row_step)
        self.last_points = None  # puntos en el suelo de la ultima imagen (para depurar)
        self.last_lines = []

    # ------------------------------------------------------------------ deteccion en la imagen

    def _line_points(self, img, T):
        """Centros de los tramos oscuros con ancho compatible con la linea -> puntos (x, y) en el suelo."""
        sub = img[self.rows].astype(np.float32)
        gray = 0.114 * sub[:, :, self.ib] + 0.587 * sub[:, :, 1] + 0.299 * sub[:, :, self.ir]
        if self.local_window > 0:
            # Fondo LOCAL (media movil a lo largo de la fila): resiste reflejos y luz no uniforme
            k = self.local_window
            pad = np.pad(gray, ((0, 0), (k // 2 + 1, k // 2)), mode="edge")
            cs = np.cumsum(pad, axis=1)
            bg = (cs[:, k:] - cs[:, :-k]) / float(k)
        else:
            bg = np.median(gray, axis=1)[:, None]
        # Contraste minimo relativo al brillo del fondo (funciona con poca y mucha luz)
        thr = np.maximum(self.min_contrast, self.rel_contrast * bg)
        if self.polaridad == "oscura":
            dark = (gray < bg * self.dark_ratio) & ((bg - gray) > thr)
        else:  # linea clara: el contraste se mide contra la escala que le queda por encima al fondo
            dark = (gray > bg + (255.0 - bg) * (1.0 - self.dark_ratio)) & ((gray - bg) > thr)
        nr, w = dark.shape
        padded = np.zeros((nr, w + 2), dtype=np.int8)
        padded[:, 1:-1] = dark
        diff = np.diff(padded, axis=1)
        r_s, start = np.nonzero(diff == 1)
        r_e, end = np.nonzero(diff == -1)
        if start.size == 0:
            return np.zeros((0, 2))
        width = end - start
        center_u = (start + end - 1) / 2.0
        center_v = self.rows[r_s].astype(float)
        x, y, depth, valid = self.cam.pixels_to_ground(center_u, center_v, T)
        depth = np.where(valid, depth, 1.0)
        expected = self.cam.fx * self.line_width / np.maximum(depth, 0.05)
        lo, hi = self.width_tol
        ok = (valid & (width >= np.maximum(1.0, lo * expected)) & (width <= hi * expected + 2.0)
              & (x > self.x_range[0]) & (x < self.x_range[1]) & (np.abs(y) < self.max_abs_y))
        self._uv = np.column_stack((center_u[ok], center_v[ok]))
        return np.column_stack((x[ok], y[ok]))

    # ------------------------------------------------------------------ autocorreccion de la pose

    def _refine_pose(self, pts, T, cl, cr, m):
        """Corrige pequenos errores de pitch/roll de la camara buscando la correccion que deja las dos
        lineas paralelas y separadas exactamente lane_width + line_width. Devuelve (cl, cr, m, dpitch, droll)
        o None si no mejora."""
        uv = self._uv
        tol = 0.05
        cosm = 1.0 / math.sqrt(1 + m * m)
        res_l = np.abs(pts[:, 1] - (cl + m * pts[:, 0])) * cosm
        res_r = np.abs(pts[:, 1] - (cr + m * pts[:, 0])) * cosm
        left = (res_l < tol) & (res_l <= res_r)
        right = (res_r < tol) & (res_r < res_l)
        if left.sum() < 5 or right.sum() < 5:
            return None
        sel = left | right
        uv_s, is_left = uv[sel], left[sel]
        cam = self.cam
        rays = np.column_stack((np.ones(len(uv_s)), -(uv_s[:, 0] - cam.cx) / cam.fx, -(uv_s[:, 1] - cam.cy) / cam.fy))
        R, t = T[:3, :3], T[:3, 3]
        dps, drs, Rs = [], [], []
        for dp in self.refine_pitch:
            for dr in self.refine_roll:
                cp, sp, cr_, sr = math.cos(dp), math.sin(dp), math.cos(dr), math.sin(dr)
                Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                Rx = np.array([[1, 0, 0], [0, cr_, -sr], [0, sr, cr_]])
                Rs.append(R.dot(Ry).dot(Rx))
                dps.append(dp)
                drs.append(dr)
        Rs = np.array(Rs)                                   # (K, 3, 3)
        d = np.einsum("kij,nj->kni", Rs, rays)              # (K, N, 3) rayos en el marco del robot
        dz = d[:, :, 2]
        good = np.all(dz < -1e-6, axis=1)
        s = -t[2] / np.where(dz < -1e-6, dz, -1.0)
        X = t[0] + s * d[:, :, 0]
        Y = t[1] + s * d[:, :, 1]

        def fit(mask):
            x, y = X[:, mask], Y[:, mask]
            xm, ym = x.mean(axis=1), y.mean(axis=1)
            vx = ((x - xm[:, None]) ** 2).mean(axis=1)
            mm = ((x - xm[:, None]) * (y - ym[:, None])).mean(axis=1) / np.maximum(vx, 1e-9)
            return ym - mm * xm, mm

        c_l, m_l = fit(is_left)
        c_r, m_r = fit(~is_left)
        m_avg = 0.5 * (m_l + m_r)
        sep = (c_l - c_r) / np.sqrt(1 + m_avg ** 2)
        cost = ((m_l - m_r) / 0.02) ** 2 + ((sep - self.spacing) / 0.02) ** 2
        cost = np.where(good, cost, np.inf)
        k = int(np.argmin(cost))
        if not np.isfinite(cost[k]):
            return None
        return float(c_l[k]), float(c_r[k]), float(m_avg[k]), dps[k], drs[k]

    def _red_points(self, img, T):
        sub = img[self.rows].astype(np.int32)
        b, g, r = sub[:, :, self.ib], sub[:, :, 1], sub[:, :, self.ir]
        red = ((r > self.red_min) & (r * 10 > g * int(10 * self.red_ratio_g)) & (r * 10 > b * int(10 * self.red_ratio_b))
               & ((r - np.maximum(g, b)) > self.red_min_sat))
        # La franja de meta cruza todo el carril: exigir muchos pixeles rojos en la misma fila
        row_count = red.sum(axis=1)
        red &= (row_count >= self.red_min_row_px)[:, None]
        rr, uu = np.nonzero(red)
        if rr.size < self.red_min_points:
            return None, int(rr.size)
        x, y, _, valid = self.cam.pixels_to_ground(uu.astype(float), self.rows[rr].astype(float), T)
        x = x[valid]
        if x.size < self.red_min_points:
            return None, int(x.size)
        return float(np.percentile(x, 10)), int(x.size)

    # ------------------------------------------------------------------ ajuste de rectas

    def _ransac_lines(self, pts):
        """Extrae hasta max_lines rectas y = c + m x. Devuelve lista de (c, m, n_inliers)."""
        lines = []
        remaining = pts
        for _ in range(self.max_lines):
            n = remaining.shape[0]
            if n < self.min_inliers:
                break
            i = self.rng.randint(0, n, self.ransac_iters)
            j = self.rng.randint(0, n, self.ransac_iters)
            x1, y1 = remaining[i, 0], remaining[i, 1]
            x2, y2 = remaining[j, 0], remaining[j, 1]
            dxs = x2 - x1
            good = np.abs(dxs) > 0.08
            if not np.any(good):
                break
            m = (y2[good] - y1[good]) / dxs[good]
            c = y1[good] - m * x1[good]
            keep = np.abs(m) < 1.2
            if not np.any(keep):
                break
            m, c = m[keep], c[keep]
            cosang = 1.0 / np.sqrt(1.0 + m * m)
            resid = np.abs(remaining[:, 1][None, :] - (c[:, None] + m[:, None] * remaining[:, 0][None, :])) * cosang[:, None]
            counts = (resid < self.inlier_tol).sum(axis=1)
            best = int(np.argmax(counts))
            if counts[best] < self.min_inliers:
                break
            inl = resid[best] < self.inlier_tol
            xs, ys = remaining[inl, 0], remaining[inl, 1]
            if np.ptp(xs) > 0.15:
                m_fit, c_fit = np.polyfit(xs, ys, 1)
            else:
                m_fit, c_fit = m[best], c[best]
            # recalcular inliers con la recta refinada
            inl = np.abs(remaining[:, 1] - (c_fit + m_fit * remaining[:, 0])) / math.sqrt(1 + m_fit ** 2) < self.inlier_tol
            if inl.sum() < self.min_inliers:
                break
            lines.append((float(c_fit), float(m_fit), int(inl.sum())))
            remaining = remaining[~inl]
        return lines

    def _choose(self, lines):
        """Elige las lineas del propio carril. Devuelve (c_left, c_right, m, n_lines)."""
        best = None
        for a in range(len(lines)):
            for b in range(len(lines)):
                if a == b:
                    continue
                cl, ml, nl = lines[a]
                cr, mr, nr = lines[b]
                if not (cl > 0.0 > cr):
                    continue
                if abs(ml - mr) > 0.15:
                    continue
                m = (ml * nl + mr * nr) / float(nl + nr)
                sep = (cl - cr) / math.sqrt(1 + m * m)
                err = abs(sep - self.spacing)
                if err > self.pair_tol:
                    continue
                score = nl + nr - 200.0 * err
                if best is None or score > best[0]:
                    best = (score, cl, cr, m)
        if best is not None:
            return best[1], best[2], best[3], 2
        # Una sola linea: la mas cercana al robot a cada lado, la de mayor soporte gana
        cands = [l for l in lines if abs(l[0]) < self.spacing * 1.1]
        if not cands:
            return None, None, 0.0, 0
        c, m, n = max(cands, key=lambda l: (l[2], -abs(l[0])))
        off = self.spacing * math.sqrt(1 + m * m)
        if c > 0:
            return c, c - off, m, 1
        return c + off, c, m, 1

    # ------------------------------------------------------------------ API

    def process(self, img, T):
        """img: arreglo HxWx3 uint8 (orden de color segun color_order). T: 4x4 camara->robot en el instante de la imagen."""
        est = LaneEstimate()
        pts = self._line_points(img, T)
        est.n_points = int(pts.shape[0])
        self.last_points = pts
        est.red_dist, est.red_points = self._red_points(img, T)
        lines = self._ransac_lines(pts) if pts.shape[0] >= self.min_inliers else []
        self.last_lines = lines
        cl, cr, m, n_lines = self._choose(lines)
        if n_lines == 0:
            return est
        self.last_refine = None
        if n_lines == 2 and self.refine:
            ref = self._refine_pose(pts, T, cl, cr, m)
            if ref is not None:
                cl, cr, m, dp, dr = ref
                self.last_refine = (dp, dr)
        yc = 0.5 * (cl + cr)              # centro del carril en x = 0 (marco del robot)
        cosm = 1.0 / math.sqrt(1 + m * m)
        est.valid = True
        est.n_lines = n_lines
        est.c_left, est.c_right, est.slope = cl, cr, m
        est.d = -yc * cosm                # + = robot a la izquierda del centro
        est.psi = -math.atan(m)           # + = robot girado a la izquierda respecto al carril
        est.quality = 1.0 if n_lines == 2 else 0.5
        return est
