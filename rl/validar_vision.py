# -*- coding: utf-8 -*-
"""Valida que la vision simulada funciona como se dice y que la politica de verdad la usa (Capa 1).

Uso (desde ~/nao_rl):
    PY=~/nao_rl/mujoco_playground/.venv/bin/python
    $PY validar_vision.py runs/nao_v06_s0_XXXX                  # 16 carreras por salida y por modo
    $PY validar_vision.py runs/nao_v06_s0_XXXX --carreras 32

No entrena nada: solo evalua en el mundo nominal, con el comando de la carrera (vx maxima de la generacion, sin giro).

1) Sensor: compara lo que mide la camara simulada con la verdad del simulador.
     - el carril virtual (d integrada) contra la geometria (d calculada con la posicion del robot)
     - ruido de cada cuadro (sesgo y desviacion) contra lo configurado
     - fraccion de cuadros perdidos y retraso efectivo
2) Salidas chuecas: el robot arranca corrido y/o girado respecto al carril y se mira si se recentra.
3) Sabotaje: las mismas salidas con la vision APAGADA (psi = d = 0, "creo que voy perfecto") e INVERTIDA
   (signos al reves). Si la politica usa la vision, apagada no corrige e invertida se sale.
4) Estadistica: cota superior (95 %) de la tasa real de salidas del carril, por Clopper-Pearson.

Guarda en <run>/validacion_vision/: resumen.csv, sensor.json, veredicto.txt, trayectorias.png, sensor.png.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import nao_carrera as nc  # noqa: E402
from evaluar import cargar, LARGO, MITAD_CARRIL, MITAD_PIE  # noqa: E402

# nombre, d0 (m, + = a la izquierda del centro), psi0 (grados, angulo del carril respecto al robot), dentro de lo
# entrenado (v06: +-10 cm, +-8 grados). "hacia afuera" = corrido a un lado y ademas apuntando hacia ese mismo lado.
SALIDAS = [
    ("centrada", 0.00, 0.0, True),
    ("izq_10cm", 0.10, 0.0, True),
    ("der_10cm", -0.10, 0.0, True),
    ("gira_+8", 0.00, 8.0, True),
    ("gira_-8", 0.00, -8.0, True),
    ("izq_10_afuera", 0.10, -8.0, True),
    ("der_10_afuera", -0.10, 8.0, True),
    ("izq_20cm", 0.20, 0.0, False),
    ("gira_+15", 0.00, 15.0, False),
]
MODOS = [("normal", 1.0), ("apagada", 0.0), ("invertida", -1.0)]


def cota_superior(k, n, conf=0.95):
    """Clopper-Pearson de una cola: la tasa real p mas alta que todavia es compatible con k fallos en n intentos."""
    if n == 0:
        return float("nan")
    if k >= n:
        return 1.0

    def cdf(p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if cdf(mid) > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


def hacer_validador(env, politica, pasos, vx):
    mj = env.mj_model
    sites = np.array([mj.site(p).id for p in nc.PIES])
    etq = env.etiquetas()[0]
    i_psi = jp.array([i for i, e in enumerate(etq) if e.endswith("_vis_psi")])
    i_d = jp.array([i for i, e in enumerate(etq) if e.endswith("_vis_d")])
    h0_psi, h0_d = etq.index("h0_vis_psi"), etq.index("h0_vis_d")
    cmd = jp.array([vx, 0.0])

    def sabotear(obs, factor, idx_psi, idx_d):
        s = obs["state"]
        s = s.at[idx_psi].multiply(factor).at[idx_d].multiply(factor)
        return dict(obs, state=s)

    def una(d0, psi0, factor, key):
        k_reset, k_obs, key = jax.random.split(key, 3)
        st = env.reset(k_reset)
        q = st.data.qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0]))     # robot mirando hacia +x del mundo
        data = mjx.forward(env.mjx_model, st.data.replace(qpos=q))
        z = jp.zeros(())
        info = dict(st.info, comando=cmd, latencia=jp.zeros((), jp.int32),
                    yaw0=z, rumbo_ref=psi0 + z, rumbo_real=z, carril_d=d0 + z,      # carril girado psi0 y corrido d0
                    vis_cuenta=jp.zeros((), jp.int32), vis_muestra=jp.array([psi0, d0]),
                    vis_obs=jp.array([psi0, d0]), pies_xy_ant=data.site_xpos[sites][:, :2])
        obs = env._obs(data, info, jp.zeros_like(st.obs["state"]), k_obs)
        st = st.replace(data=data, info=info, obs=sabotear(obs, factor, i_psi, i_d))
        p0 = q[0:2]
        u = jp.array([jp.cos(psi0), jp.sin(psi0)])        # direccion del carril
        n = jp.array([-jp.sin(psi0), jp.cos(psi0)])       # normal del carril (hacia la izquierda)

        def paso(carry, k):
            st, key, fin, cayo, t_meta, fuera, dmax, s_fin, d_fin, psi_fin, vis_ant = carry
            key, k_pol = jax.random.split(key)
            accion, _ = politica(st.obs, k_pol)
            st = env.step(st, accion)
            st = st.replace(info=dict(st.info, comando=cmd, paso=jp.zeros_like(st.info["paso"])),
                            obs=sabotear(st.obs, factor, h0_psi, h0_d))
            d = st.data
            rel = d.qpos[0:2] - p0
            s = jp.dot(rel, u)                                # avance a lo largo del carril
            d_geo = d0 + jp.dot(rel, n)                       # distancia al centro, por geometria
            d_pies = d0 + (d.site_xpos[sites][:, :2] - p0) @ n
            psi_real = env._error_rumbo(st.info)
            vivo = ~fin
            t = (k + 1) * env.dt
            llega = (s >= LARGO) & vivo
            cae = (st.done > 0) & vivo
            t_meta = jp.where(llega, t, t_meta)
            cayo = cayo | cae
            fuera = fuera | (vivo & (jp.max(jp.abs(d_pies)) + MITAD_PIE > MITAD_CARRIL))
            dmax = jp.where(vivo, jp.maximum(dmax, jp.abs(d_geo)), dmax)
            s_fin = jp.where(vivo, s, s_fin)
            d_fin = jp.where(vivo, d_geo, d_fin)
            psi_fin = jp.where(vivo, psi_real, psi_fin)
            fin = fin | llega | cae
            traza = {"vivo": vivo, "s": s, "d_geo": d_geo, "d_int": st.info["carril_d"], "psi": psi_real,
                     "vis_obs": st.info["vis_obs"], "vis_ant": vis_ant, "vis_muestra": st.info["vis_muestra"],
                     "cuenta": st.info["vis_cuenta"]}
            carry = (st, key, fin, cayo, t_meta, fuera, dmax, s_fin, d_fin, psi_fin, st.info["vis_obs"])
            return carry, traza

        f = jp.array(False)
        c0 = (st, key, f, f, jp.array(jp.nan), f, jp.abs(d0) + z, z, d0 + z, psi0 + z, st.info["vis_obs"])
        c, traza = jax.lax.scan(paso, c0, jp.arange(pasos))
        res = {"cayo": c[3], "t_meta": c[4], "fuera": c[5], "d_max": c[6], "s_fin": c[7], "d_fin": c[8],
               "psi_fin": c[9]}
        return res, traza

    return jax.jit(jax.vmap(una, in_axes=(0, 0, None, 0)))


def revisar_sensor(env, traza):
    """Estadisticas de la camara simulada (solo modo normal, solo mientras el robot esta en carrera)."""
    v = env._config.vision
    vivo = traza["vivo"]
    out = {}
    out["carril_int_vs_geo_max_cm"] = float(100 * np.max(np.abs(traza["d_int"] - traza["d_geo"])[vivo]))
    # cuadro recien tomado: la cuenta se acaba de reiniciar a periodo-1 en ese paso
    toma = vivo & (traza["cuenta"] == v.periodo - 1)
    verdad = np.stack([traza["psi"], traza["d_int"]], -1)
    err = traza["vis_muestra"] - verdad
    err[..., 0] = np.arctan2(np.sin(err[..., 0]), np.cos(err[..., 0]))
    e_psi, e_d = err[..., 0][toma], err[..., 1][toma]
    out.update(cuadros=int(toma.sum()),
               ruido_psi_media_deg=float(np.degrees(e_psi.mean())), ruido_psi_sigma_deg=float(np.degrees(e_psi.std())),
               ruido_psi_config_deg=float(np.degrees(v.sigma_psi)),
               ruido_d_media_cm=float(100 * e_d.mean()), ruido_d_sigma_cm=float(100 * e_d.std()),
               ruido_d_config_cm=float(100 * v.sigma_d))
    # perdida: en un paso de toma, la observacion deberia cambiar (se publica el cuadro anterior); si no, se perdio
    toma_pub = toma.copy()
    toma_pub[:, 0] = False                       # el primer cuadro publica lo mismo que ya habia (salida perfecta)
    igual = np.all(traza["vis_obs"] == traza["vis_ant"], axis=-1)
    out["perdida_medida"] = float(igual[toma_pub].mean())
    out["perdida_config"] = float(v.prob_perdida)
    # retraso efectivo: el corrimiento L (en pasos) que mejor alinea lo observado con la verdad
    obs_psi, ver_psi = traza["vis_obs"][..., 0], traza["psi"]
    errores = []
    for L in range(0, 4 * v.periodo + 1):
        m = vivo[:, L:] & vivo[:, :vivo.shape[1] - L]
        dif = obs_psi[:, L:] - ver_psi[:, :ver_psi.shape[1] - L]
        errores.append(float(np.sqrt(np.mean(np.square(dif[m])))))
    L = int(np.argmin(errores))
    out["retraso_medido_ms"] = 1000 * L * env.dt
    out["retraso_esperado_ms"] = 1000 * (1.5 * v.periodo - 0.5) * env.dt   # 1 cuadro de retraso + medio periodo
    out["rmse_por_retraso_deg"] = [round(math.degrees(e), 3) for e in errores]
    return out


def veredictos(sensor, filas, env):
    v = env._config.vision
    lin = []

    def chequeo(ok, texto):
        lin.append(("OK   " if ok else "FALLA") + "  " + texto)
        return ok

    s = sensor
    chequeo(s["carril_int_vs_geo_max_cm"] < 1.0,
            "carril virtual coherente: d integrada vs geometria, error max %.2f cm (< 1 cm)" % s["carril_int_vs_geo_max_cm"])
    for nom, med, sig, conf, u in (("psi", s["ruido_psi_media_deg"], s["ruido_psi_sigma_deg"], s["ruido_psi_config_deg"], "grados"),
                                   ("d", s["ruido_d_media_cm"], s["ruido_d_sigma_cm"], s["ruido_d_config_cm"], "cm")):
        chequeo(abs(sig / conf - 1) < 0.15 and abs(med) < 0.25 * conf,
                "ruido de %s: sesgo %.3f, sigma %.3f %s (configurado %.3f)" % (nom, med, sig, u, conf))
    chequeo(abs(s["perdida_medida"] - s["perdida_config"]) < 0.02,
            "cuadros perdidos: %.1f %% (configurado %.1f %%)" % (100 * s["perdida_medida"], 100 * s["perdida_config"]))
    chequeo(abs(s["retraso_medido_ms"] - s["retraso_esperado_ms"]) <= 2 * 1000 * env.dt + 1e-6,
            "retraso efectivo: %.0f ms (esperado ~%.0f ms)" % (s["retraso_medido_ms"], s["retraso_esperado_ms"]))

    def grupo(modo, cond):
        return [f for f in filas if f["modo"] == modo and cond(f)]

    chuecas = lambda f: f["dentro"] and f["salida"] != "centrada"   # noqa: E731
    err = {m: np.mean([f["d_fin_abs_cm"] for f in grupo(m, chuecas)]) for m, _ in MODOS}
    sal = {m: sum(f["salidas"] for f in grupo(m, chuecas)) for m, _ in MODOS}
    chequeo(err["normal"] < 0.5 * err["apagada"] and sal["normal"] <= sal["apagada"],
            "usa la vision: error lateral final %.1f cm (normal) vs %.1f cm (apagada); salidas %d vs %d"
            % (err["normal"], err["apagada"], sal["normal"], sal["apagada"]))
    chequeo(err["invertida"] > err["normal"] and sal["invertida"] >= sal["normal"],
            "invertir la vision la empeora: %.1f cm y %d salidas (normal %.1f cm, %d)"
            % (err["invertida"], sal["invertida"], err["normal"], sal["normal"]))
    for f in grupo("normal", lambda f: f["dentro"] and f["salida"] != "centrada"):
        if f["d0_cm"] != 0 and f["psi0_deg"] == 0:
            chequeo(f["d_fin_cm"] * np.sign(f["d0_cm"]) < 0.5 * abs(f["d0_cm"]),
                    "%s: se recentra hacia el lado correcto (%.1f -> %.1f cm)" % (f["salida"], f["d0_cm"], f["d_fin_cm"]))
        if f["psi0_deg"] != 0 and f["d0_cm"] == 0:
            chequeo(abs(f["psi_fin_deg"]) < 0.5 * abs(f["psi0_deg"]),
                    "%s: corrige el rumbo (%.1f -> %.1f grados)" % (f["salida"], f["psi0_deg"], f["psi_fin_deg"]))
    norm = grupo("normal", lambda f: f["dentro"])
    k, n = sum(f["salidas"] for f in norm), sum(f["carreras"] for f in norm)
    kc, nc_ = sum(f["carreras"] - f["metas"] for f in norm), n
    lin.append("")
    lin.append("Estadistica (modo normal, salidas dentro de lo entrenado, %d carreras):" % n)
    lin.append("  salidas del carril: %d/%d -> la tasa real es <= %.1f %% (95 %%)" % (k, n, 100 * cota_superior(k, n)))
    lin.append("  no llega a la meta: %d/%d -> la tasa real es <= %.1f %% (95 %%)" % (kc, nc_, 100 * cota_superior(kc, nc_)))
    return lin


def graficar(carpeta, filas_traza, env):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(sin matplotlib: no se hacen las graficas)")
        return
    fig, ejes = plt.subplots(1, len(MODOS), figsize=(16, 5), sharey=True)
    colores = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax, (modo, _) in zip(ejes, MODOS):
        for j, (nombre, d0, psi0, dentro) in enumerate(SALIDAS):
            tr = filas_traza[(modo, nombre)]
            for r in range(tr["s"].shape[0]):
                m = tr["vivo"][r]
                ax.plot(tr["s"][r][m], 100 * tr["d_geo"][r][m], color=colores[j], lw=0.7, alpha=0.6,
                        ls="-" if dentro else "--", label=nombre if r == 0 else None)
        for y in (-100 * MITAD_CARRIL, 100 * MITAD_CARRIL):
            ax.axhline(y, color="k", lw=1.5)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlim(0, LARGO)
        ax.set_ylim(-60, 60)
        ax.set_title("vision %s" % modo)
        ax.set_xlabel("avance a lo largo del carril (m)")
    ejes[0].set_ylabel("distancia al centro del carril (cm, + = izquierda)")
    ejes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Salidas chuecas: la linea negra es el borde del carril (70 cm). Punteadas: fuera de lo entrenado")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "trayectorias.png"), dpi=120)
    plt.close(fig)

    tr = filas_traza[("normal", "izq_10_afuera")]
    m = tr["vivo"][0]
    t = env.dt * np.arange(m.size)[m]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(t, 100 * tr["d_int"][0][m], label="d real")
    a1.step(t, 100 * tr["vis_obs"][0][m][:, 1], where="post", label="d que ve la politica")
    a1.set_ylabel("d (cm)")
    a1.legend()
    a2.plot(t, np.degrees(tr["psi"][0][m]), label="psi real")
    a2.step(t, np.degrees(tr["vis_obs"][0][m][:, 0]), where="post", label="psi que ve la politica")
    a2.set_ylabel("psi (grados)")
    a2.set_xlabel("tiempo (s)")
    a2.legend()
    fig.suptitle("Camara simulada en una carrera (salida izq_10_afuera): ruido, 10 Hz, retraso y cuadros perdidos")
    fig.tight_layout()
    fig.savefig(os.path.join(carpeta, "sensor.png"), dpi=120)
    plt.close(fig)


def validar(run, carreras=16, segundos=30.0):
    t0 = time.time()
    env, politica, cfg = cargar(run)
    if not env._config.vision.activa:
        sys.exit("%s no tiene vision (vision.activa = False): no hay nada que validar" % run)
    vx = float(env._config.vx[1])
    pasos = int(round(segundos / env.dt))
    val = hacer_validador(env, politica, pasos, vx)
    d0 = jp.array(np.repeat([s[1] for s in SALIDAS], carreras), jp.float32)
    psi0 = jp.array(np.repeat([math.radians(s[2]) for s in SALIDAS], carreras), jp.float32)
    claves = jax.random.split(jax.random.PRNGKey(1000), len(SALIDAS) * carreras)

    filas, trazas, sensor = [], {}, None
    for modo, factor in MODOS:
        res, traza = val(d0, psi0, jp.float32(factor), claves)
        res = {k: np.asarray(v) for k, v in res.items()}
        traza = {k: np.asarray(v) for k, v in traza.items()}
        if modo == "normal":
            sensor = revisar_sensor(env, traza)
        for j, (nombre, sd0, spsi0, dentro) in enumerate(SALIDAS):
            sl = slice(j * carreras, (j + 1) * carreras)
            r = {k: v[sl] for k, v in res.items()}
            trazas[(modo, nombre)] = {k: v[sl] for k, v in traza.items()}
            metas = np.isfinite(r["t_meta"])
            filas.append({
                "modo": modo, "salida": nombre, "d0_cm": 100 * sd0, "psi0_deg": spsi0, "dentro": dentro,
                "carreras": carreras, "metas": int(metas.sum()), "caidas": int(r["cayo"].sum()),
                "salidas": int(r["fuera"].sum()),
                "t_meta_medio": float(np.nanmean(r["t_meta"])) if metas.any() else float("nan"),
                "d_fin_cm": float(100 * r["d_fin"].mean()), "d_fin_abs_cm": float(100 * np.abs(r["d_fin"]).mean()),
                "d_max_cm": float(100 * r["d_max"].mean()), "psi_fin_deg": float(np.degrees(r["psi_fin"].mean())),
                "avance_medio": float(r["s_fin"].mean()),
            })

    carpeta = os.path.join(run, "validacion_vision")
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "resumen.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    json.dump(sensor, open(os.path.join(carpeta, "sensor.json"), "w"), indent=1)
    lineas = veredictos(sensor, filas, env)
    open(os.path.join(carpeta, "veredicto.txt"), "w").write("\n".join(lineas) + "\n")
    graficar(carpeta, trazas, env)

    print("\n%s  validacion de la vision (%d carreras por salida, %.1f m/s)  %.1f min"
          % (os.path.basename(run.rstrip("/")), carreras, vx, (time.time() - t0) / 60))
    print("%-10s %-14s %6s %6s %6s %8s %9s %9s %9s" % ("modo", "salida", "metas", "caidas", "sale", "t meta",
                                                       "d0 cm", "d fin cm", "|d| max"))
    for f_ in filas:
        print("%-10s %-14s %3d/%-2d %6d %3d/%-2d %8s %9.1f %9.1f %9.1f%s" % (
            f_["modo"], f_["salida"], f_["metas"], carreras, f_["caidas"], f_["salidas"], carreras,
            "-" if math.isnan(f_["t_meta_medio"]) else "%.1f s" % f_["t_meta_medio"],
            f_["d0_cm"], f_["d_fin_cm"], f_["d_max_cm"], "" if f_["dentro"] else "  (no entrenada)"))
    print("\nVeredicto:")
    print("\n".join("  " + x for x in lineas))
    print("\nArchivos en", carpeta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="una o varias carpetas runs/... de politicas con vision")
    ap.add_argument("--carreras", type=int, default=16)
    ap.add_argument("--segundos", type=float, default=30.0)
    a = ap.parse_args()
    for run in a.runs:
        validar(run, a.carreras, a.segundos)


if __name__ == "__main__":
    main()
