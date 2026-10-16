dash.calc('peor', {
    inputs: ['motores'],
    title: 'Motor de pierna más exigido',
    description: 'El motor de pierna con mayor calor (par RMS ÷ par continuo) entre todas las políticas, y cuántos pasan de 80 %.',
    fn: (motores) => {
      const piernas = motores.filter((r) => r.calor !== null && r.calor !== undefined);
      const m = piernas.reduce((a, b) => (b.calor > a.calor ? b : a));
      return [{ motor: m.motor, politica: m.politica, calor_pct: Math.round(m.calor * 100),
                sobre_80: piernas.filter((r) => r.calor > 0.8).length, total: piernas.length }];
    },
  });
  dash.calc('rapida', {
    inputs: ['resumen'],
    title: 'Política más rápida',
    description: 'La política con menor tiempo medio en 5 m.',
    fn: (resumen) => {
      const m = resumen.reduce((a, b) => (b.t_5m < a.t_5m ? b : a));
      return [{ politica: m.politica, t_5m: m.t_5m }];
    },
  });
  dash.calc('rodilla', {
    inputs: ['motores'],
    title: 'Rodilla en el tope simulado',
    description: 'La rodilla que más tiempo pasa en el tope de par del motor simulado, en porcentaje.',
    fn: (motores) => {
      const m = motores.filter((r) => r.articulacion === 'KneePitch').reduce((a, b) => (b.tope_sim > a.tope_sim ? b : a));
      return [{ politica: m.politica, motor: m.motor, tope_pct: Math.round(m.tope_sim * 100) }];
    },
  });
  dash.calc('impacto', {
    inputs: ['resumen'],
    title: 'Golpe máximo del pie',
    description: 'La mayor fuerza pico de un pie contra el piso, en veces el peso del robot.',
    fn: (resumen) => {
      const m = resumen.reduce((a, b) => (b.impacto_pico > a.impacto_pico ? b : a));
      return [{ politica: m.politica, impacto_pico: m.impacto_pico }];
    },
  });

(function () {
    const NOMBRE = {
      HipYawPitch: 'Cadera yaw', HipRoll: 'Cadera roll', HipPitch: 'Cadera pitch', KneePitch: 'Rodilla',
      AnklePitch: 'Tobillo pitch', AnkleRoll: 'Tobillo roll', ShoulderPitch: 'Hombro', ElbowRoll: 'Codo',
    };
    const PIERNA = ['LHipYawPitch', 'LHipRoll', 'RHipRoll', 'LHipPitch', 'RHipPitch', 'LKneePitch', 'RKneePitch',
                    'LAnklePitch', 'RAnklePitch', 'LAnkleRoll', 'RAnkleRoll'];
    const etiqueta = (m) => NOMBRE[m.slice(1)] + (m === 'LHipYawPitch' ? '' : (m[0] === 'L' ? ' izq.' : ' der.'));
    const pct0 = d3.format('.0%'), pct1 = d3.format('.1%'), f2 = d3.format('.2f'), f1 = d3.format('.1f');
    const css = (v) => getComputedStyle(document.getElementById('dash-root')).getPropertyValue(v).trim();
    const sel = (s) => document.querySelector(s);

    function fallo(el) { el.replaceChildren(); const d = document.createElement('div'); d.className = 'fail'; d.textContent = 'Failed to load'; el.appendChild(d); }
    function cargando(el, ok) { el.classList.toggle('dash-skeleton', !ok); }

    function tooltip(box) {
      let t = box.querySelector('.tip');
      if (!t) { t = document.createElement('div'); t.className = 'tip'; t.hidden = true; box.appendChild(t); }
      return t;
    }
    function ponerTip(t, box, x, y, filas, titulo) {
      t.replaceChildren();
      const h = document.createElement('div'); h.className = 'h'; h.textContent = titulo; t.appendChild(h);
      filas.forEach(([color, nombre, valor]) => {
        const r = document.createElement('div');
        if (color) { const s = document.createElement('span'); s.className = 'sw'; s.style.background = color; r.appendChild(s); }
        r.appendChild(document.createTextNode(nombre + '  ' + valor)); t.appendChild(r);
      });
      t.hidden = false;
      const w = t.offsetWidth, bw = box.clientWidth;
      t.style.left = Math.max(0, Math.min(bw - w, x + 12)) + 'px';
      t.style.top = Math.max(0, y - 10) + 'px';
    }
    function medir(svg, texto, clase) {
      const n = svg.append('text').attr('class', clase || '').attr('font-size', 12).text(texto);
      const w = n.node().getComputedTextLength(); n.remove(); return w;
    }

    // ---------------- mapa de calor
    function dibujarHeat(mot, res) {
      const box = sel('#heat');
      const svgOld = box.querySelector('svg'); if (svgOld) svgOld.remove();
      const W = box.clientWidth, pols = res.map((r) => r.politica);
      const svg = d3.select(box).append('svg').attr('width', '100%');
      const mL = Math.max(...pols.map((p) => medir(svg, p))) + 12, mT = 46, rowH = 34;
      const H = mT + pols.length * rowH + 58;
      svg.attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
      const x = d3.scaleBand(PIERNA, [mL, W]).padding(0.06), y = d3.scaleBand(pols, [mT, mT + pols.length * rowH]).padding(0.08);
      // escala termica fija (se lee igual en tema claro y oscuro): frio = poco esfuerzo, caliente = mucho
      const PARADAS = [0.2, 0.4, 0.55, 0.65, 0.75, 0.8];
      const COLORES = ['#e3eefa', '#a9cbef', '#f7e08a', '#f4b04d', '#e8742f', '#c62828'];
      const escala = d3.scaleLinear(PARADAS, COLORES).interpolate(d3.interpolateRgb).clamp(true);
      const color = (v) => (v > 0.8 ? '#b71c1c' : escala(v));
      const textoSobre = (v) => { const c = d3.rgb(color(v)); const l = (0.299 * c.r + 0.587 * c.g + 0.114 * c.b) / 255; return l > 0.6 ? '#1b1b1b' : '#ffffff'; };
      const narrow = x.bandwidth() < 46;
      svg.append('g').selectAll('text').data(PIERNA).join('text')
        .attr('x', (d) => x(d) + x.bandwidth() / 2).attr('y', mT - 8).attr('text-anchor', narrow ? 'start' : 'middle')
        .attr('transform', (d) => (narrow ? `rotate(-40 ${x(d) + x.bandwidth() / 2} ${mT - 8})` : null))
        .attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((d) => (narrow ? etiqueta(d).replace('Tobillo', 'Tob.').replace('Cadera', 'Cad.') : etiqueta(d)));
      if (!narrow) svg.selectAll('g text').each(function (d) { // dos lineas cuando hay espacio
        const n = d3.select(this), e = etiqueta(d).split(' ');
        n.text(null); n.append('tspan').attr('x', x(d) + x.bandwidth() / 2).attr('dy', e.length > 2 ? '-1.1em' : '-0.1em').text(e.slice(0, 2).join(' '));
        if (e.length > 2) n.append('tspan').attr('x', x(d) + x.bandwidth() / 2).attr('dy', '1.15em').text(e.slice(2).join(' '));
      });
      svg.append('g').selectAll('text').data(pols).join('text')
        .attr('x', mL - 10).attr('y', (d) => y(d) + y.bandwidth() / 2).attr('dy', '0.35em').attr('text-anchor', 'end')
        .attr('font-size', 12).attr('fill', 'var(--color-fg)').text((d) => d);
      // leyenda: barra de color de 20 % a 100 %
      const lx0 = mL, lx1 = Math.min(W - 10, mL + 360), ly = mT + pols.length * rowH + 20;
      const lx = d3.scaleLinear([0.2, 1.0], [lx0, lx1]);
      const gid = 'grad-calor-' + Math.round(W);
      const grad = svg.append('defs').append('linearGradient').attr('id', gid).attr('x1', 0).attr('x2', 1).attr('y1', 0).attr('y2', 0);
      d3.range(0.2, 1.0001, 0.025).forEach((v) => grad.append('stop').attr('offset', (v - 0.2) / 0.8).style('stop-color', color(v)));
      svg.append('rect').attr('x', lx0).attr('y', ly).attr('width', lx1 - lx0).attr('height', 10).attr('rx', 3).attr('fill', `url(#${gid})`);
      svg.append('line').attr('x1', lx(0.8)).attr('x2', lx(0.8)).attr('y1', ly - 4).attr('y2', ly + 14).attr('stroke', 'var(--color-fg)').attr('stroke-width', 1.5);
      svg.append('g').selectAll('text').data([0.2, 0.4, 0.6, 0.8, 1.0]).join('text').attr('x', (d) => lx(d)).attr('y', ly + 26)
        .attr('text-anchor', 'middle').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((d) => pct0(d));
      svg.append('text').attr('x', lx1 + 8).attr('y', ly + 9).attr('font-size', 11).attr('fill', 'var(--color-fg-muted)')
        .text(lx1 + 120 < W ? 'límite 80 %' : '');
      const celdas = mot.filter((r) => PIERNA.includes(r.motor));
      const tip = tooltip(box);
      const g = svg.append('g').selectAll('g').data(celdas).join('g');
      g.append('rect').attr('x', (d) => x(d.motor)).attr('y', (d) => y(d.politica)).attr('width', x.bandwidth()).attr('height', y.bandwidth())
        .attr('rx', 4).attr('fill', (d) => color(d.calor))
        .on('mousemove', (ev, d) => { const [px, py] = d3.pointer(ev, box);
          ponerTip(tip, box, px, py, [[null, 'Calor', pct0(d.calor)], [null, 'Sobre el continuo', pct0(d.sobre_continuo)], [null, 'En el tope sim.', pct0(d.tope_sim)]], d.politica + ' · ' + etiqueta(d.motor)); })
        .on('mouseleave', () => { tip.hidden = true; });
      if (x.bandwidth() >= 30) g.append('text').attr('x', (d) => x(d.motor) + x.bandwidth() / 2).attr('y', (d) => y(d.politica) + y.bandwidth() / 2)
        .attr('dy', '0.35em').attr('text-anchor', 'middle').attr('font-size', 11).attr('pointer-events', 'none')
        .attr('fill', (d) => textoSobre(d.calor)).text((d) => Math.round(d.calor * 100));
    }

    // ---------------- serie de par
    function dibujarSerie(serie, pol) {
      const box = sel('#serie-chart');
      const svgOld = box.querySelector('svg'); if (svgOld) svgOld.remove();
      const filas = serie.filter((r) => r.politica === pol);
      const motores = ['LHipPitch', 'LKneePitch', 'LAnklePitch'];
      const W = box.clientWidth, H = 260, mT = 10, mB = 28, mL = 40;
      const svg = d3.select(box).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
      const mR = Math.max(...motores.map((m) => medir(svg, etiqueta(m)))) + 14;
      const x = d3.scaleLinear([3, 7], [mL, W - mR]), y = d3.scaleLinear([-5.6, 5.6], [H - mB, mT]);
      svg.append('g').selectAll('line').data(y.ticks(6)).join('line').attr('x1', mL).attr('x2', W - mR)
        .attr('y1', (d) => y(d)).attr('y2', (d) => y(d)).attr('stroke', 'var(--cds-chart-grid)');
      [2.33, -2.33].forEach((v) => svg.append('line').attr('x1', mL).attr('x2', W - mR).attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', 'var(--cds-chart-reference)').attr('stroke-dasharray', '5 4'));
      [5, -5].forEach((v) => svg.append('line').attr('x1', mL).attr('x2', W - mR).attr('y1', y(v)).attr('y2', y(v))
        .attr('stroke', 'var(--cds-chart-status-critical)').attr('stroke-width', 1).attr('opacity', 0.7));
      const ax = svg.append('g').attr('transform', `translate(0,${H - mB})`).call(d3.axisBottom(x).ticks(5).tickFormat((d) => d + ' s'));
      const ay = svg.append('g').attr('transform', `translate(${mL},0)`).call(d3.axisLeft(y).ticks(6));
      [ax, ay].forEach((a) => { a.attr('font-size', null).attr('font-family', null); a.selectAll('text').attr('fill', 'var(--color-fg-muted)').attr('font-size', 11); a.selectAll('line,path').attr('stroke', 'var(--cds-chart-axis)'); });
      svg.append('text').attr('x', mL).attr('y', mT - 0).attr('dy', '-0.2em').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text('N·m');
      const linea = d3.line().x((d) => x(d.t)).y((d) => y(d.par));
      const porMotor = motores.map((m) => filas.filter((r) => r.motor === m));
      porMotor.forEach((datos, i) => {
        svg.append('path').datum(datos).attr('fill', 'none').attr('stroke', dash.colors[i]).attr('stroke-width', 1.6)
          .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round').attr('d', linea);
      });
      // etiquetas al final, separadas
      const fin = porMotor.map((d, i) => ({ i, y: d.length ? y(d[d.length - 1].par) : 0 })).sort((a, b) => a.y - b.y);
      for (let k = 1; k < fin.length; k++) if (fin[k].y - fin[k - 1].y < 14) fin[k].y = fin[k - 1].y + 14;
      fin.forEach((f) => svg.append('text').attr('x', W - mR + 6).attr('y', f.y).attr('dy', '0.35em').attr('font-size', 12)
        .attr('fill', dash.colors[f.i]).text(etiqueta(motores[f.i])));
      const tip = tooltip(box), regla = svg.append('line').attr('y1', mT).attr('y2', H - mB).attr('stroke', 'var(--cds-chart-axis)').attr('opacity', 0);
      svg.append('rect').attr('x', mL).attr('y', mT).attr('width', W - mL - mR).attr('height', H - mT - mB).attr('fill', 'transparent')
        .on('mousemove', (ev) => {
          const [px, py] = d3.pointer(ev, box), t = x.invert(d3.pointer(ev)[0]);
          const i = d3.bisector((d) => d.t).center(porMotor[0], t);
          if (!porMotor[0][i]) return;
          regla.attr('x1', x(porMotor[0][i].t)).attr('x2', x(porMotor[0][i].t)).attr('opacity', 1);
          ponerTip(tip, box, px, py, porMotor.map((d, k) => [dash.colors[k], etiqueta(motores[k]), d[i] ? f2(d[i].par) + ' N·m' : '—']), f2(porMotor[0][i].t) + ' s');
        })
        .on('mouseleave', () => { tip.hidden = true; regla.attr('opacity', 0); });
    }

    // ---------------- barras horizontales
    function barras(boxSel, datos, valor, max, ref, fmt, colorFn) {
      const box = sel(boxSel);
      const svgOld = box.querySelector('svg'); if (svgOld) svgOld.remove();
      const W = box.clientWidth, filaH = 22, mT = 8, mB = 22;
      const H = mT + datos.length * filaH + mB;
      const svg = d3.select(box).append('svg').attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`).attr('height', H);
      const mL = Math.max(...datos.map((d) => medir(svg, etiqueta(d.motor)))) + 10, mR = 44;
      const x = d3.scaleLinear([0, max], [mL, W - mR]), y = d3.scaleBand(datos.map((d) => d.motor), [mT, H - mB]).padding(0.25);
      svg.append('g').selectAll('line').data(x.ticks(4)).join('line').attr('x1', (d) => x(d)).attr('x2', (d) => x(d))
        .attr('y1', mT).attr('y2', H - mB).attr('stroke', 'var(--cds-chart-grid)');
      svg.append('g').selectAll('text').data(x.ticks(4)).join('text').attr('x', (d) => x(d)).attr('y', H - 6).attr('text-anchor', 'middle')
        .attr('font-size', 11).attr('fill', 'var(--color-fg-muted)').text((d) => pct0(d));
      if (ref !== null) svg.append('line').attr('x1', x(ref)).attr('x2', x(ref)).attr('y1', mT - 4).attr('y2', H - mB)
        .attr('stroke', 'var(--cds-chart-status-critical)').attr('stroke-dasharray', '4 3');
      svg.append('g').selectAll('text').data(datos).join('text').attr('x', mL - 8).attr('y', (d) => y(d.motor) + y.bandwidth() / 2)
        .attr('dy', '0.35em').attr('text-anchor', 'end').attr('font-size', 12).attr('fill', 'var(--color-fg)').text((d) => etiqueta(d.motor));
      svg.append('g').selectAll('rect').data(datos).join('rect').attr('x', mL).attr('y', (d) => y(d.motor))
        .attr('width', (d) => Math.max(0, x(Math.min(max, valor(d))) - mL)).attr('height', y.bandwidth()).attr('rx', 4).attr('fill', colorFn);
      svg.append('g').selectAll('text').data(datos).join('text').attr('x', (d) => x(Math.min(max, valor(d))) + 5)
        .attr('y', (d) => y(d.motor) + y.bandwidth() / 2).attr('dy', '0.35em').attr('font-size', 11).attr('fill', 'var(--color-fg-muted)')
        .text((d) => fmt(valor(d)));
    }

    // ---------------- tablas
    function tabla(id, filas, columnas, clave) {
      const tb = sel(id + ' tbody');
      tb.replaceChildren(...filas.map((r) => {
        const tr = document.createElement('tr');
        tr.dataset.key = clave(r);
        columnas.forEach(([campo, fmt, clase]) => {
          const td = document.createElement('td');
          const v = r[campo];
          td.textContent = v === null || v === undefined ? '—' : (fmt ? fmt(v) : String(v));
          if (clase) { const c = clase(r); if (c) td.className = c; }
          tr.appendChild(td);
        });
        return tr;
      }));
    }

    // ---------------- selector de politica
    const nav = sel('#selector');
    nav.addEventListener('click', (ev) => {
      const b = ev.target.closest('button[data-pol]');
      if (b) { dash.setParams({ pol: b.dataset.pol }); dash.setLink('detalle'); }
    });

    const enc = (s) => encodeURIComponent(s);
    let primera = true;
    dash.onData(() => {
      const R = dash.data('resumen'), M = dash.data('motores'), S = dash.data('serie'), F = dash.data('referencia');
      const okR = R.status === 'ok', okM = M.status === 'ok', okS = S.status === 'ok';
      const pol = dash.params().pol || 'v06';

      ['#heat', '#tope-chart', '#vel-chart'].forEach((s) => cargando(sel(s), okM));
      cargando(sel('#serie-chart'), okS);
      document.querySelectorAll('.dash-skeleton').forEach((el) => {
        if (el.dataset.source === 'peor' || el.dataset.source === 'rodilla') cargando(el, okM);
        if (el.dataset.source === 'rapida' || el.dataset.source === 'impacto') cargando(el, okR);
      });

      // calculos de la cabecera y las cifras
      [['peor', ['motor', 'politica', 'calor_pct', 'sobre_80', 'total']], ['rapida', ['t_5m', 'politica']],
       ['rodilla', ['tope_pct', 'motor', 'politica']], ['impacto', ['impacto_pico', 'politica']]].forEach(([id, campos]) => {
        const d = dash.data(id);
        document.querySelectorAll(`[data-source="${id}"]`).forEach((el) => {
          const f = el.dataset.field;
          if (d.status !== 'ok' || !d.data.length) { el.textContent = '—'; return; }
          const v = d.data[0][f];
          el.textContent = f === 't_5m' || f === 'impacto_pico' ? f2(v) : String(v);
        });
      });

      // selector
      if (okR) {
        nav.replaceChildren(...R.data.map((r) => {
          const b = document.createElement('button'); b.type = 'button'; b.dataset.pol = r.politica;
          b.setAttribute('aria-pressed', String(r.politica === pol)); b.textContent = r.nombre; return b;
        }));
        const fila = R.data.find((r) => r.politica === pol);
        [['#lp-t', 't_5m', f2], ['#lp-e', 'energia_j_m', f1], ['#lp-i', 'impacto_p99', f2], ['#lp-s', 'separacion_min_cm', f1]].forEach(([s, campo, fmt]) => {
          const el = sel(s); el.dataset.where = 'politica=' + enc(pol); el.textContent = fila ? fmt(fila[campo]) : '—';
        });
      } else {
        nav.replaceChildren();
        ['#lp-t', '#lp-e', '#lp-i', '#lp-s'].forEach((s) => { sel(s).textContent = '—'; });
      }

      // mapa de calor
      if (okM && okR) dibujarHeat(M.data, R.data);
      else if (M.status === 'error' || R.status === 'error') fallo(sel('#heat'));
      else sel('#heat').replaceChildren();

      // serie
      if (okS) dibujarSerie(S.data, pol);
      else if (S.status === 'error') fallo(sel('#serie-chart')); else sel('#serie-chart').replaceChildren();

      // barras y tabla de motores
      if (okM) {
        const filas = M.data.filter((r) => r.politica === pol);
        const tope = [...filas].sort((a, b) => b.tope_sim - a.tope_sim);
        barras('#tope-chart', tope, (d) => d.tope_sim, Math.max(0.25, d3.max(tope, (d) => d.tope_sim) * 1.1), null, pct1,
          (d) => (d.tope_sim > 0.15 ? 'var(--cds-chart-status-critical)' : dash.colors[0]));
        const vel = [...filas].sort((a, b) => b.uso_vel - a.uso_vel);
        barras('#vel-chart', vel, (d) => d.uso_vel, 1.2, 1.0, pct0,
          (d) => (d.uso_vel > 1 ? 'var(--cds-chart-status-critical)' : dash.colors[1]));
        const alto = (campo, a, c) => (r) => (r[campo] === null ? '' : r[campo] > c ? 'critico' : r[campo] > a ? 'alto' : '');
        tabla('#tm', filas, [
          ['motor', null], ['tipo', null], ['calor', pct0, alto('calor', 0.7, 0.8)], ['sobre_continuo', pct0],
          ['tope_sim', pct1, alto('tope_sim', 0.08, 0.15)], ['fuera_curva', pct1, alto('fuera_curva', 0.002, 0.01)],
          ['vel_p99', f2], ['w_max', f2], ['cerca_tope', pct1, alto('cerca_tope', 0.01, 0.05)], ['par_rms', f2], ['par_max', f2],
        ], (r) => enc(r.politica) + ',' + enc(r.motor));
      } else {
        ['#tope-chart', '#vel-chart'].forEach((s) => (M.status === 'error' ? fallo(sel(s)) : sel(s).replaceChildren()));
        sel('#tm tbody').replaceChildren();
      }

      if (okR) tabla('#tp', R.data, [
        ['politica', null], ['vx', f1], ['vision', null], ['t_5m', f2],
        ['metas', null], ['energia_j_m', f1], ['calor_max', pct0, (r) => (r.calor_max > 0.8 ? 'critico' : r.calor_max > 0.7 ? 'alto' : '')],
        ['motor_mas_caliente', null], ['tope_rodilla', pct1], ['impacto_pico', f2], ['separacion_min_cm', f1],
      ], (r) => r.politica);
      else sel('#tp tbody').replaceChildren();

      if (F.status === 'ok') tabla('#tref', F.data, [
        ['tipo', null], ['articulaciones', null], ['reductora', (v) => String(v)], ['par_parada', f2],
        ['par_continuo', (v) => f2(v)], ['w_max', f2],
      ], (r) => enc(r.tipo));
      else sel('#tref tbody').replaceChildren();

      if (primera && okR && okM) {
        primera = false;
        const named = document.getElementById(location.hash.slice(1));
        if (named) { named.scrollIntoView(); }
      }
    });
  })();
