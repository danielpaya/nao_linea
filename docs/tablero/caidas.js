// Seccion "Caidas": carga los resultados de demo/caidas.py (tablero/datos/caidas_*.json) y dibuja graficas y tablas.
// Tambien maneja las pestanas Esfuerzo / Caidas.
(function () {
  const ARCHIVOS = ['resumen', 'articulaciones', 'causas', 'piezas', 'lista'];
  const D = {};
  const est = { caso: 'activo', pol: 'todas', metrica: 'sobrecarga' };
  let ORDEN_POL = [];            // politicas en el orden de los datos (primero las de Daniel, luego las de Camilo)
  const FUENTE = {};
  const pct0 = d3.format('.0%'), f2 = d3.format('.2f'), f1 = d3.format('.1f');
  const $ = (s) => document.querySelector(s);
  let listo = false;

  // ---------------- pestanas
  const pags = { esfuerzo: $('#pag-esfuerzo'), caidas: $('#pag-caidas') };
  function mostrar(nombre) {
    Object.entries(pags).forEach(([k, el]) => { el.hidden = k !== nombre; });
    document.querySelectorAll('.pestanas button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.pag === nombre)));
    try { history.replaceState(null, '', nombre === 'caidas' ? '#caidas' : location.pathname); } catch (e) { /* sin historia */ }
    if (nombre === 'caidas') dibujar(); else if (window.dash && dash.redibujar) dash.redibujar();
  }
  document.querySelectorAll('.pestanas button').forEach((b) => b.addEventListener('click', () => mostrar(b.dataset.pag)));

  // ---------------- datos
  Promise.all(ARCHIVOS.map((n) => fetch('tablero/datos/caidas_' + n + '.json').then((r) => {
    if (!r.ok) throw new Error(n + ': ' + r.status); return r.json();
  }))).then((xs) => {
    ARCHIVOS.forEach((n, i) => { D[n] = xs[i]; });
    D.resumen.forEach((r) => { if (r.politica !== 'todas' && !(r.politica in FUENTE)) { FUENTE[r.politica] = r.fuente || 'daniel'; ORDEN_POL.push(r.politica); } });
    listo = true;
    armarControles();
    if (location.hash === '#caidas') mostrar('caidas'); else if (!pags.caidas.hidden) dibujar();
  }).catch((e) => {
    console.error(e);
    $('#ca-conclusion').textContent = 'No se pudieron cargar los datos de caídas (' + e.message + ').';
  });
  if (location.hash === '#caidas') mostrar('caidas');

  function armarControles() {
    const pols = ['todas'].concat(ORDEN_POL.filter((p) => D.resumen.some((r) => r.politica === p)));
    const cont = $('#ca-pol');
    cont.replaceChildren(...pols.map((p) => {
      const b = document.createElement('button'); b.type = 'button'; b.dataset.pol = p;
      b.textContent = p === 'todas' ? 'Todas juntas' : p;
      if (p !== 'todas') b.title = FUENTE[p] === 'camilo' ? 'Política de Camilo' : 'Política de Daniel';
      return b;
    }));
    cont.addEventListener('click', (ev) => { const b = ev.target.closest('button[data-pol]'); if (b) { est.pol = b.dataset.pol; dibujar(); } });
    $('#ca-caso').addEventListener('click', (ev) => { const b = ev.target.closest('button[data-caso]'); if (b) { est.caso = b.dataset.caso; dibujar(); } });
    $('#ca-metrica').addEventListener('click', (ev) => { const b = ev.target.closest('button[data-m]'); if (b) { est.metrica = b.dataset.m; dibujar(); } });
  }

  function marcar(selector, attr, valor) {
    document.querySelectorAll(selector + ' button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset[attr] === valor)));
  }

  // ---------------- utilidades de dibujo
  function lienzo(sel, alto) {
    const box = $(sel); const viejo = box.querySelector('svg'); if (viejo) viejo.remove();
    const W = box.clientWidth;
    const svg = d3.select(box).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${alto}`).attr('height', alto);
    return { box, svg, W };
  }
  function medir(svg, t, tam) { const n = svg.append('text').attr('font-size', tam || 12).text(t); const w = n.node().getComputedTextLength(); n.remove(); return w; }
  function tip(box) { let t = box.querySelector('.tip'); if (!t) { t = document.createElement('div'); t.className = 'tip'; t.hidden = true; box.appendChild(t); } return t; }
  function ponerTip(t, box, x, y, titulo, filas) {
    t.replaceChildren(); const h = document.createElement('div'); h.className = 'h'; h.textContent = titulo; t.appendChild(h);
    filas.forEach((f) => { const r = document.createElement('div'); r.textContent = f; t.appendChild(r); });
    t.hidden = false; const w = t.offsetWidth; t.style.left = Math.max(0, Math.min(box.clientWidth - w, x + 12)) + 'px'; t.style.top = Math.max(0, y - 10) + 'px';
  }
  const ESCALA = d3.scaleLinear([0, 0.15, 0.35, 0.55, 0.75, 1.0], ['#e3eefa', '#a9cbef', '#f7e08a', '#f4b04d', '#e8742f', '#c62828'])
    .interpolate(d3.interpolateRgb).clamp(true);
  const textoSobre = (c) => { const r = d3.rgb(c); return (0.299 * r.r + 0.587 * r.g + 0.114 * r.b) / 255 > 0.6 ? '#1b1b1b' : '#ffffff'; };

  // ---------------- dibujo general
  function dibujar() {
    if (!listo || pags.caidas.hidden) return;
    marcar('#ca-caso', 'caso', est.caso); marcar('#ca-pol', 'pol', est.pol); marcar('#ca-metrica', 'm', est.metrica);
    const R = D.resumen.find((r) => r.politica === est.pol && r.caso === est.caso);
    const T = D.resumen.find((r) => r.politica === 'todas' && r.caso === 'activo');
    const Tp = D.resumen.find((r) => r.politica === 'todas' && r.caso === 'protegido');
    const artT = D.articulaciones.filter((a) => a.politica === 'todas' && a.caso === 'activo' && a.carga_rel_mediana !== null);
    const peor = artT.reduce((a, b) => (b.carga_rel_mediana > a.carga_rel_mediana ? b : a));
    const porPol = D.resumen.filter((r) => r.politica !== 'todas' && r.caso === 'activo');
    const nDan = porPol.filter((r) => (r.fuente || 'daniel') === 'daniel').length, nCam = porPol.length - nDan;
    const peorPierna = artT.filter((a) => a.grupo === 'pierna').reduce((a, b) => (b.sobrecarga > a.sobrecarga ? b : a));
    const causasT = D.causas.filter((c) => c.politica === 'todas' && c.caso === 'activo');
    const peorCausa = causasT.reduce((a, b) => (b.severidad_mediana > a.severidad_mediana ? b : a));
    $('#ca-n-caidas').textContent = d3.min(porPol, (r) => r.caidas);
    $('#ca-conclusion').textContent =
      `En ${T.caidas} caídas (${porPol.length} políticas${nCam ? `: ${nDan} de Daniel y ${nCam} de Camilo` : ''}), la articulación más ` +
      `cargada frente a su motor es ${peor.nombre}: carga típica de ${f1(peor.carga_rel_mediana)} veces su par de parada; una mano toca el ` +
      `piso primero en ${pct0(T.manos_primero)} de las caídas. En las piernas, ${pct0(T.sobrecarga_pierna)} de las caídas sobrecargan alguna ` +
      `articulación con los motores encendidos (la más frecuente: ${peorPierna.nombre.toLowerCase()}, ${pct0(peorPierna.sobrecarga)}) y ` +
      `${pct0(Tp.sobrecarga_pierna)} con los motores sueltos. La causa más severa es: ${peorCausa.nombre.toLowerCase()}.`;
    $('#ca-t-caidas').textContent = R.caidas;
    $('#ca-t-caidas-n').textContent = `de ${R.simulaciones} simulaciones; en ${R.resistio} el robot aguantó la causa sin caerse`;
    $('#ca-t-sobre').textContent = pct0(R.sobrecarga_pierna);
    $('#ca-t-sev').textContent = f2(R.severidad_mediana) + ' ×';
    $('#ca-t-sev-n').textContent = `el par de parada · en el 5 % peor: ${f2(R.severidad_p95)} ×`;
    $('#ca-t-cabeza').textContent = pct0(R.cabeza_primero);
    $('#ca-t-cabeza-n').textContent = `golpe típico de la cabeza: ${f1(R.cabeza_impacto_mediana)} × el peso del robot`;
    dibujarHeat(); dibujarBarras(); dibujarPiezas(R); dibujarDirecciones(R); tablaCausas(); tablaComparacion(); tablaLista();
  }

  // ---------------- mapa: articulaciones x politicas
  function dibujarHeat() {
    const m = est.metrica;
    const notas = {
      sobrecarga: 'Porcentaje de caídas en que la carga sobre la articulación pasa el par de parada de su motor.',
      sobrevelocidad: 'Porcentaje de caídas en que la articulación es arrastrada más rápido que la velocidad máxima de su motor.',
      golpe_tope: 'Porcentaje de caídas en que la articulación golpea su tope mecánico (más de 2 N·m contra el tope).',
    };
    $('#ca-mapa-nota').textContent = notas[m] + ' Caso ' + (est.caso === 'activo' ? 'activo (motores encendidos).' : 'protegido (motores sueltos).');
    const filas = D.articulaciones.filter((a) => a.politica === 'todas' && a.caso === est.caso && (m === 'golpe_tope' ? true : a[m] !== null));
    const arts = filas.map((a) => a.articulacion);
    const nombres = Object.fromEntries(filas.map((a) => [a.articulacion, a.nombre]));
    const pols = ORDEN_POL.filter((p) => D.resumen.some((r) => r.politica === p)).concat(['todas']);
    const rowH = 24;
    const largo = d3.max(pols, (p) => (p === 'todas' ? 5 : p.length));
    const angosto = ($('#ca-heat').clientWidth - 150) / pols.length < 7.2 * largo + 10;   // si no caben, inclinados
    const mT = angosto ? 14 + 6.2 * largo : 30;
    const H = mT + arts.length * rowH + 8;
    const { box, svg, W } = lienzo('#ca-heat', H);
    const mL = Math.max(...arts.map((a) => medir(svg, nombres[a]))) + 12;
    const x = d3.scaleBand(pols, [mL, W]).padding(0.06), y = d3.scaleBand(arts, [mT, H - 8]).padding(0.08);
    svg.append('g').selectAll('text').data(pols).join('text')
      .attr('transform', (p) => `translate(${x(p) + x.bandwidth() / 2},${mT - (angosto ? 6 : 10)})` + (angosto ? ' rotate(-45)' : ''))
      .attr('text-anchor', angosto ? 'start' : 'middle').attr('font-size', angosto ? 11 : 12).attr('fill', 'var(--color-fg-muted)')
      .text((p) => (p === 'todas' ? 'Todas' : p));
    svg.append('g').selectAll('text').data(arts).join('text').attr('x', mL - 8).attr('y', (a) => y(a) + y.bandwidth() / 2).attr('dy', '0.35em')
      .attr('text-anchor', 'end').attr('font-size', 12).attr('fill', 'var(--color-fg)').text((a) => nombres[a]);
    const celdas = D.articulaciones.filter((a) => a.caso === est.caso && arts.includes(a.articulacion) && pols.includes(a.politica));
    const t = tip(box);
    const g = svg.append('g').selectAll('g').data(celdas).join('g');
    g.append('rect').attr('x', (d) => x(d.politica)).attr('y', (d) => y(d.articulacion)).attr('width', x.bandwidth()).attr('height', y.bandwidth())
      .attr('rx', 4).attr('fill', (d) => ESCALA(d[m] || 0))
      .on('mousemove', (ev, d) => { const [px, py] = d3.pointer(ev, box);
        ponerTip(t, box, px, py, (d.politica === 'todas' ? 'Todas' : d.politica) + ' · ' + d.nombre, [
          'Sobrecarga: ' + (d.sobrecarga === null ? 'sin dato de motor' : pct0(d.sobrecarga)),
          'Sobrevelocidad: ' + (d.sobrevelocidad === null ? 'sin dato de motor' : pct0(d.sobrevelocidad)),
          'Golpe contra el tope: ' + pct0(d.golpe_tope),
          'Carga mediana: ' + f1(d.carga_mediana) + ' N·m' + (d.parada ? ' (parada ' + d.parada + ')' : '')]); })
      .on('mouseleave', () => { t.hidden = true; });
    if (x.bandwidth() > 34) g.append('text').attr('x', (d) => x(d.politica) + x.bandwidth() / 2).attr('y', (d) => y(d.articulacion) + y.bandwidth() / 2)
      .attr('dy', '0.35em').attr('text-anchor', 'middle').attr('font-size', 11).attr('pointer-events', 'none')
      .attr('fill', (d) => textoSobre(ESCALA(d[m] || 0))).text((d) => Math.round(100 * (d[m] || 0)));
  }

  // ---------------- barras de carga relativa
  function dibujarBarras() {
    const filas = D.articulaciones.filter((a) => a.politica === est.pol && a.caso === est.caso && a.parada)
      .map((a) => ({ ...a, p95r: a.carga_p95 / a.parada })).sort((a, b) => b.carga_rel_mediana - a.carga_rel_mediana);
    const filaH = 24, mT = 8, mB = 24, H = mT + filas.length * filaH + mB;
    const { box, svg, W } = lienzo('#ca-barras', H);
    const mL = Math.max(...filas.map((a) => medir(svg, a.nombre))) + 10, mR = 50;
    const max = Math.max(2, d3.max(filas, (a) => a.p95r) * 1.05);
    const x = d3.scaleLinear([0, max], [mL, W - mR]), y = d3.scaleBand(filas.map((a) => a.articulacion), [mT, H - mB]).padding(0.28);
    svg.append('g').selectAll('line').data(x.ticks(5)).join('line').attr('x1', (d) => x(d)).attr('x2', (d) => x(d)).attr('y1', mT).attr('y2', H - mB).attr('stroke', 'var(--cds-chart-grid)');
    svg.append('g').selectAll('text').data(x.ticks(5)).join('text').attr('x', (d) => x(d)).attr('y', H - 6).attr('text-anchor', 'middle')
      .attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((d) => d + '×');
    svg.append('line').attr('x1', x(1)).attr('x2', x(1)).attr('y1', mT - 4).attr('y2', H - mB).attr('stroke', 'var(--cds-chart-status-critical)').attr('stroke-dasharray', '4 3');
    svg.append('g').selectAll('text').data(filas).join('text').attr('x', mL - 8).attr('y', (a) => y(a.articulacion) + y.bandwidth() / 2).attr('dy', '0.35em')
      .attr('text-anchor', 'end').attr('font-size', 12).attr('fill', 'var(--color-fg)').text((a) => a.nombre);
    svg.append('g').selectAll('line').data(filas).join('line').attr('x1', mL).attr('x2', (a) => x(Math.min(max, a.p95r)))
      .attr('y1', (a) => y(a.articulacion) + y.bandwidth() / 2).attr('y2', (a) => y(a.articulacion) + y.bandwidth() / 2).attr('stroke', 'var(--cds-chart-reference)').attr('stroke-width', 2);
    const t = tip(box);
    svg.append('g').selectAll('rect').data(filas).join('rect').attr('x', mL).attr('y', (a) => y(a.articulacion))
      .attr('width', (a) => Math.max(0, x(Math.min(max, a.carga_rel_mediana)) - mL)).attr('height', y.bandwidth()).attr('rx', 4)
      .attr('fill', (a) => (a.carga_rel_mediana > 1 ? '#c62828' : a.carga_rel_mediana > 0.7 ? '#e8742f' : dash.colors[0]))
      .on('mousemove', (ev, a) => { const [px, py] = d3.pointer(ev, box);
        ponerTip(t, box, px, py, a.nombre, ['Carga mediana: ' + f1(a.carga_mediana) + ' N·m (' + f2(a.carga_rel_mediana) + '×)',
          '95 % de las caídas: ' + f1(a.carga_p95) + ' N·m (' + f2(a.p95r) + '×)', 'Par de parada del motor: ' + a.parada + ' N·m']); })
      .on('mouseleave', () => { t.hidden = true; });
    svg.append('g').selectAll('text.v').data(filas).join('text').attr('x', (a) => x(Math.min(max, a.p95r)) + 6)
      .attr('y', (a) => y(a.articulacion) + y.bandwidth() / 2).attr('dy', '0.35em').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)')
      .text((a) => f1(a.carga_rel_mediana) + '×');
  }

  // ---------------- piezas que tocan primero
  function dibujarPiezas() {
    const filas = D.piezas.filter((p) => p.politica === est.pol && p.caso === est.caso && !p.pieza.endsWith('foot') && p.primera > 0)
      .sort((a, b) => b.primera - a.primera).slice(0, 8);
    const filaH = 26, mT = 8, mB = 22, H = mT + Math.max(filas.length, 1) * filaH + mB;
    const { box, svg, W } = lienzo('#ca-piezas-chart', H);
    const mL = Math.max(...filas.map((p) => medir(svg, p.nombre)), 40) + 10, mR = 110;
    const x = d3.scaleLinear([0, Math.max(0.5, d3.max(filas, (p) => p.primera))], [mL, W - mR]), y = d3.scaleBand(filas.map((p) => p.pieza), [mT, H - mB]).padding(0.28);
    svg.append('g').selectAll('text').data(x.ticks(4)).join('text').attr('x', (d) => x(d)).attr('y', H - 6).attr('text-anchor', 'middle')
      .attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((d) => pct0(d));
    svg.append('g').selectAll('text').data(filas).join('text').attr('x', mL - 8).attr('y', (p) => y(p.pieza) + y.bandwidth() / 2).attr('dy', '0.35em')
      .attr('text-anchor', 'end').attr('font-size', 12).attr('fill', 'var(--color-fg)').text((p) => p.nombre);
    svg.append('g').selectAll('rect').data(filas).join('rect').attr('x', mL).attr('y', (p) => y(p.pieza))
      .attr('width', (p) => Math.max(0, x(p.primera) - mL)).attr('height', y.bandwidth()).attr('rx', 4)
      .attr('fill', (p) => (p.pieza === 'head' ? '#c62828' : dash.colors[2]));
    svg.append('g').selectAll('text.v').data(filas).join('text').attr('x', (p) => x(p.primera) + 6).attr('y', (p) => y(p.pieza) + y.bandwidth() / 2)
      .attr('dy', '0.35em').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((p) => pct0(p.primera) + ' · golpe ' + f1(p.impacto_mediana) + '× peso');
  }

  // ---------------- direcciones
  function dibujarDirecciones(R) {
    const dirs = [['adelante', 'Adelante'], ['atras', 'Atrás'], ['izquierda', 'Izquierda'], ['derecha', 'Derecha']];
    const tot = d3.sum(dirs, ([k]) => R['dir_' + k]) || 1;
    const filas = dirs.map(([k, n]) => ({ k, n, v: R['dir_' + k] / tot }));
    const filaH = 34, mT = 8, mB = 22, H = mT + filas.length * filaH + mB;
    const { svg, W } = lienzo('#ca-dir-chart', H);
    const mL = Math.max(...filas.map((f) => medir(svg, f.n))) + 10, mR = 50;
    const x = d3.scaleLinear([0, Math.max(0.5, d3.max(filas, (f) => f.v))], [mL, W - mR]), y = d3.scaleBand(filas.map((f) => f.k), [mT, H - mB]).padding(0.3);
    svg.append('g').selectAll('text').data(x.ticks(4)).join('text').attr('x', (d) => x(d)).attr('y', H - 6).attr('text-anchor', 'middle')
      .attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((d) => pct0(d));
    svg.append('g').selectAll('text').data(filas).join('text').attr('x', mL - 8).attr('y', (f) => y(f.k) + y.bandwidth() / 2).attr('dy', '0.35em')
      .attr('text-anchor', 'end').attr('font-size', 12).attr('fill', 'var(--color-fg)').text((f) => f.n);
    svg.append('g').selectAll('rect').data(filas).join('rect').attr('x', mL).attr('y', (f) => y(f.k)).attr('width', (f) => Math.max(0, x(f.v) - mL))
      .attr('height', y.bandwidth()).attr('rx', 4).attr('fill', dash.colors[1]);
    svg.append('g').selectAll('text.v').data(filas).join('text').attr('x', (f) => x(f.v) + 6).attr('y', (f) => y(f.k) + y.bandwidth() / 2)
      .attr('dy', '0.35em').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((f) => pct0(f.v));
  }

  // ---------------- tablas
  function llenar(sel, filas, celdas) {
    $(sel + ' tbody').replaceChildren(...filas.map((r) => {
      const tr = document.createElement('tr');
      celdas(r).forEach(([texto, clase]) => { const td = document.createElement('td'); td.textContent = texto; if (clase) td.className = clase; tr.appendChild(td); });
      return tr;
    }));
  }
  const nivel = (v, a, c) => (v > c ? 'critico' : v > a ? 'alto' : '');
  const DIRS = { adelante: 'adelante', atras: 'atrás', izquierda: 'izquierda', derecha: 'derecha' };
  function tablaCausas() {
    const filas = D.causas.filter((c) => c.politica === est.pol && c.caso === est.caso).sort((a, b) => b.severidad_mediana - a.severidad_mediana);
    llenar('#ca-tcausas', filas, (c) => [[c.nombre], [String(c.caidas)], [f2(c.severidad_mediana) + '×', nivel(c.severidad_mediana, 0.9, 1)],
      [f2(c.severidad_p95) + '×', nivel(c.severidad_p95, 1.5, 2)], [pct0(c.sobrecarga_pierna), nivel(c.sobrecarga_pierna, 0.5, 0.8)],
      [f2(c.t_caer_mediana) + ' s'], [c.peor_articulacion], [c.primera_pieza], [DIRS[c.direccion] || c.direccion]]);
  }
  function tablaComparacion() {
    const a = D.articulaciones.filter((x) => x.politica === est.pol && x.caso === 'activo');
    const p = Object.fromEntries(D.articulaciones.filter((x) => x.politica === est.pol && x.caso === 'protegido').map((x) => [x.articulacion, x]));
    const filas = a.filter((x) => x.parada || x.golpe_tope > 0.02 || p[x.articulacion].golpe_tope > 0.02);
    const fmt = (v) => (v === null ? '—' : pct0(v));
    llenar('#ca-tcomp', filas, (x) => {
      const q = p[x.articulacion];
      return [[x.nombre], [fmt(x.sobrecarga), x.sobrecarga === null ? '' : nivel(x.sobrecarga, 0.3, 0.5)], [fmt(q.sobrecarga), q.sobrecarga === null ? '' : nivel(q.sobrecarga, 0.3, 0.5)],
        [fmt(x.golpe_tope), nivel(x.golpe_tope, 0.25, 0.5)], [fmt(q.golpe_tope), nivel(q.golpe_tope, 0.25, 0.5)],
        [fmt(x.sobrevelocidad), x.sobrevelocidad === null ? '' : nivel(x.sobrevelocidad, 0.4, 0.7)], [fmt(q.sobrevelocidad), q.sobrevelocidad === null ? '' : nivel(q.sobrevelocidad, 0.4, 0.7)]];
    });
  }
  function tablaLista() {
    const pol = est.pol === 'todas' ? ORDEN_POL[0] : est.pol;
    const c = est.caso;
    $('#ca-lista h2').textContent = 'Todas las caídas de ' + pol + (est.pol === 'todas' ? ' (elige una política para ver las suyas)' : '');
    const filas = D.lista.filter((r) => r.politica === pol);
    llenar('#ca-tlista', filas, (r) => [[String(r.id + 1)], [r.causa + (r.nivel ? ' (×' + (1.5 ** r.nivel).toFixed(2).replace(/0+$/, '').replace(/\.$/, '') + ')' : '')], [r.detalle],
      [DIRS[r.direccion] || r.direccion], [f2(r.t_caer)], [r['primera_' + c] || '—'], [f2(r['sev_' + c]) + '×', nivel(r['sev_' + c], 0.9, 1)],
      [r['peor_' + c]], [f2(r['peor_rel_' + c]) + '×', nivel(r['peor_rel_' + c], 1, 2)]]);
  }

  let ancho = window.innerWidth, t = null;
  window.addEventListener('resize', () => { if (window.innerWidth === ancho) return; ancho = window.innerWidth; clearTimeout(t); t = setTimeout(dibujar, 150); });
})();
