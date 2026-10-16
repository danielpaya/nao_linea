// Reemplazo minimo de la API "dash" del tablero de Claude, para que la misma pagina corra sola en GitHub Pages.
// Carga los JSON de tablero/datos/, calcula los dash.calc y vuelve a dibujar con dash.onData cuando llegan los
// datos, cambia el ancho o se elige otra politica.
(function () {
  const FUENTES = {
    resumen: 'tablero/datos/resumen.json',
    motores: 'tablero/datos/motores.json',
    serie: 'tablero/datos/serie.json',
    referencia: 'tablero/datos/referencia.json',
  };
  const DEFECTO = { pol: 'v06' };
  const estado = {};
  const calcs = {};
  const oyentes = [];
  let params = Object.assign({}, DEFECTO);

  function avisar() {
    oyentes.forEach((fn) => {
      try { fn(); } catch (e) { console.error(e); }
    });
  }

  function marcar() {
    document.querySelectorAll('[data-source]').forEach((el) => {
      const ids = el.dataset.source.split(/\s+/);
      const st = ids.map((id) => dash.data(id).status);
      const s = st.includes('error') ? 'error' : st.includes('loading') ? 'loading' : null;
      if (s) el.setAttribute('data-dash-source-state', s); else el.removeAttribute('data-dash-source-state');
    });
  }

  const dash = {
    colors: ['--serie-1', '--serie-2', '--serie-3', '--serie-4', '--serie-5', '--serie-6', '--serie-7', '--serie-8']
      .map((v) => `var(${v})`),
    data(id) {
      if (calcs[id]) {
        const c = calcs[id];
        const ins = c.inputs.map((i) => dash.data(i));
        if (ins.some((d) => d.status === 'error')) return { status: 'error', data: [], message: 'Falta un dato de entrada' };
        if (ins.some((d) => d.status !== 'ok')) return { status: 'loading', data: [] };
        try {
          const r = c.fn(...ins.map((d) => d.data));
          return { status: 'ok', data: r, rows: r, meta: [] };
        } catch (e) {
          return { status: 'error', data: [], message: String(e) };
        }
      }
      const e = estado[id];
      if (!e) return { status: 'missing', data: [] };
      return e;
    },
    onData(fn) { oyentes.push(fn); try { fn(); } catch (e) { console.error(e); } },
    calc(id, def) { calcs[id] = def; },
    loader() {},
    refresh() {},
    params() { return Object.assign({}, params); },
    setParams(p) { params = Object.assign({}, params, p); avisar(); },
    resetParams() { params = Object.assign({}, DEFECTO); avisar(); },
    setLink(nombre) { try { history.replaceState(null, '', '#' + nombre); } catch (e) { /* sin historia */ } },
  };
  window.dash = dash;

  Object.entries(FUENTES).forEach(([id, url]) => {
    estado[id] = { status: 'loading', data: [], meta: [] };
    fetch(url)
      .then((r) => { if (!r.ok) throw new Error(r.status + ' ' + r.statusText); return r.json(); })
      .then((filas) => {
        estado[id] = { status: 'ok', data: filas, rows: filas, columns: filas.length ? Object.keys(filas[0]) : [], meta: [] };
      })
      .catch((err) => { estado[id] = { status: 'error', data: [], message: String(err) }; console.error(id, err); })
      .finally(() => { marcar(); avisar(); });
  });

  let ancho = window.innerWidth, t = null;
  window.addEventListener('resize', () => {
    if (window.innerWidth === ancho) return;
    ancho = window.innerWidth;
    clearTimeout(t);
    t = setTimeout(avisar, 150);
  });

  // tema claro / oscuro (el boton cambia entre los dos; por defecto sigue al sistema)
  document.addEventListener('DOMContentLoaded', () => {
    const b = document.getElementById('tema');
    if (!b) return;
    const oscuro = () => document.documentElement.dataset.theme === 'dark' ||
      (!document.documentElement.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
    const rotulo = () => { b.textContent = oscuro() ? 'Tema claro' : 'Tema oscuro'; };
    rotulo();
    b.addEventListener('click', () => { document.documentElement.dataset.theme = oscuro() ? 'light' : 'dark'; rotulo(); avisar(); });
  });
})();
