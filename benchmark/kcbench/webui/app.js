/* kcbench console ---------------------------------------------------------
   Talks only to the Flask endpoints in webview.py. Every number it draws is
   read back out of a run file that an ordinary cb.py command wrote. */

'use strict';

// ── i18n ──────────────────────────────────────────────────────────────────
const I18N = {
  en: {
    'brand.sub': 'benchmark console',
    'run.start': 'Run', 'run.stop': 'Stop', 'run.extra': 'extra flags',
    'tab.inputs': 'Input', 'tab.settings': 'Settings',
    'tab.outputs': 'Output', 'tab.runs': 'Runs',
    'cfg.save': 'Save',
    'cfg.hint': 'Saved to config.json, with the previous version kept as .bak. Takes effect on the next command; restart the webview to re-resolve its own paths.',
    'cfg.roots': 'Resolved paths',
    'runs.dashboard': 'Dashboard', 'runs.refresh': 'Refresh',
    'log.title': 'Log', 'log.follow': 'Follow', 'log.clear': 'Clear',
    'viewer.empty': 'Nothing selected',
    'viewer.hintTitle': 'Pick a file on either side, or open the dashboard.',
    'viewer.hint1': 'Left — the corpus and the generated data a command reads.',
    'viewer.hint2': 'Right — what it wrote: item sets, run files, charts.',
    'viewer.hint3': 'A run file opens as a scored dashboard, not raw JSON.',
    'st.idle': 'idle', 'st.running': 'running', 'st.done': 'done', 'st.failed': 'failed',
    'st.stopped': 'stopped',
    'v.records': 'records', 'v.showing': 'showing', 'v.of': 'of',
    'v.prev': 'Previous', 'v.next': 'Next', 'v.raw': 'Raw JSON', 'v.back': 'Back',
    'v.open': 'Open full size', 'v.clipped': 'clipped for display',
    'v.truncated': 'summarised — the file is too large to send whole',
    'v.binary': 'Binary file. Nothing to show.',
    'v.empty': 'This directory is empty.',
    'd.title': 'Scored runs', 'd.none': 'No run files yet. Score a model and they appear here.',
    'd.note': 'Every bar is read out of a run file. Bars share a 0–1 scale unless the metric says otherwise; perplexity is plotted on its own scale and lower is better.',
    'd.metric': 'Metric', 'd.compare': 'compare',
    'd.cmpTitle': 'Paired comparisons', 'd.cmpNote': 'Written by cb.py compare. A p-value near 1 means the data cannot tell the change from noise.',
    'd.sig': 'significant', 'd.noise': 'noise',
    'd.headline': 'Headline', 'd.select': 'Tick two or more runs to chart them together.',
    'r.stopped': 'stop requested', 'r.busy': 'A command is already running.',
    'e.saved': 'Saved.', 'e.required': 'is required',
  },
  ko: {
    'brand.sub': '벤치마크 콘솔',
    'run.start': '실행', 'run.stop': '중지', 'run.extra': '추가 플래그',
    'tab.inputs': '입력', 'tab.settings': '설정',
    'tab.outputs': '출력', 'tab.runs': '실행 결과',
    'cfg.save': '저장',
    'cfg.hint': 'config.json에 저장하고 이전 내용은 .bak으로 남깁니다. 다음 명령부터 적용되며, 웹뷰 자체의 경로를 다시 잡으려면 재시작하십시오.',
    'cfg.roots': '해석된 경로',
    'runs.dashboard': '대시보드', 'runs.refresh': '새로고침',
    'log.title': '로그', 'log.follow': '자동 스크롤', 'log.clear': '지우기',
    'viewer.empty': '선택된 항목 없음',
    'viewer.hintTitle': '좌우 패널에서 파일을 고르거나 대시보드를 여십시오.',
    'viewer.hint1': '왼쪽 — 명령이 읽는 코퍼스와 생성 데이터.',
    'viewer.hint2': '오른쪽 — 명령이 쓴 것: 문항 세트, 실행 결과 파일, 차트.',
    'viewer.hint3': '실행 결과 파일은 원본 JSON이 아니라 채점 대시보드로 열립니다.',
    'st.idle': '대기', 'st.running': '실행 중', 'st.done': '완료', 'st.failed': '실패',
    'st.stopped': '중지됨',
    'v.records': '레코드', 'v.showing': '표시', 'v.of': '/',
    'v.prev': '이전', 'v.next': '다음', 'v.raw': '원본 JSON', 'v.back': '뒤로',
    'v.open': '원본 크기로 열기', 'v.clipped': '표시용으로 잘림',
    'v.truncated': '파일이 너무 커서 구조만 요약했습니다',
    'v.binary': '바이너리 파일입니다. 표시할 내용이 없습니다.',
    'v.empty': '빈 디렉터리입니다.',
    'd.title': '채점된 실행', 'd.none': '아직 실행 결과가 없습니다. 모델을 채점하면 여기에 나타납니다.',
    'd.note': '모든 막대는 실행 결과 파일에서 읽은 값입니다. 별도 표기가 없으면 0–1 척도를 공유하며, perplexity는 자체 척도로 그리고 낮을수록 좋습니다.',
    'd.metric': '지표', 'd.compare': '비교',
    'd.cmpTitle': '짝지은 비교', 'd.cmpNote': 'cb.py compare가 쓴 값입니다. p값이 1에 가까우면 변화를 잡음과 구별할 수 없다는 뜻입니다.',
    'd.sig': '유의미', 'd.noise': '잡음',
    'd.headline': '대표 지표', 'd.select': '두 개 이상 선택하면 함께 그립니다.',
    'r.stopped': '중지 요청됨', 'r.busy': '이미 실행 중인 명령이 있습니다.',
    'e.saved': '저장했습니다.', 'e.required': ' 값이 필요합니다',
  },
};

let LANG = localStorage.getItem('kc.lang') || 'en';
const t = (k) => (I18N[LANG] && I18N[LANG][k]) || I18N.en[k] || k;

function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    el.textContent = t(el.dataset.i18n);
  });
  document.querySelectorAll('[data-i18n-ph]').forEach((el) => {
    el.placeholder = t(el.dataset.i18nPh);
  });
  $('lang').textContent = LANG === 'ko' ? '한' : 'EN';
  renderStatus();
  if (STATE.commands) renderFields();
  if (VIEW.reload) VIEW.reload();
}

// ── small helpers ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmtSize = (b) => {
  if (b === undefined || b === null) return '';
  if (b < 1024) return b + ' B';
  if (b < 1024 * 1024) return (b / 1024).toFixed(0) + ' K';
  return (b / 1048576).toFixed(1) + ' M';
};
const fmtNum = (v) => (typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(2) : v.toFixed(3)) : String(v));
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw Object.assign(new Error(body.error || `HTTP ${r.status}`), { body });
  return body;
}

const STATE = { commands: null, roots: {}, job: null, runs: [] };
const VIEW = { reload: null };

// ── theme ─────────────────────────────────────────────────────────────────
function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  localStorage.setItem('kc.theme', mode);
  $('theme').textContent = mode === 'dark' ? '◑' : '◐';
  if (VIEW.reload) VIEW.reload();      // canvases are drawn with theme colours
}

// ── command panel ─────────────────────────────────────────────────────────
function renderCommands() {
  const sel = $('cmd');
  sel.innerHTML = '';
  const groups = {};
  Object.entries(STATE.commands).forEach(([name, spec]) => {
    (groups[spec.group] ||= []).push(name);
  });
  Object.entries(groups).forEach(([group, names]) => {
    const og = el('optgroup');
    og.label = group;
    names.forEach((n) => {
      const o = el('option', null, n);
      o.value = n;
      og.appendChild(o);
    });
    sel.appendChild(og);
  });
  sel.value = localStorage.getItem('kc.cmd') || 'eval';
  if (!sel.value) sel.value = 'build';
}

function renderFields() {
  const spec = STATE.commands[$('cmd').value];
  const box = $('fields');
  box.innerHTML = '';
  (spec.fields || []).forEach((f) => {
    const wrap = el('div', 'field' + (f.type === 'flag' ? ' flag' : ''));
    const id = 'f_' + f.flag.replace(/-/g, '_');
    const label = el('label', null, f.label || f.flag);
    label.htmlFor = id;
    if (f.help) wrap.title = f.help;

    let input;
    if (f.type === 'flag') {
      input = el('input');
      input.type = 'checkbox';
      wrap.append(input, label);
    } else if (f.type === 'select') {
      input = el('select');
      (f.options || []).forEach((o) => {
        const opt = el('option', null, o || '—');
        opt.value = o;
        input.appendChild(opt);
      });
      wrap.append(label, input);
    } else if (f.type === 'run') {
      input = el('input');
      input.type = 'text';
      input.setAttribute('list', 'runtags');
      wrap.classList.add('wide');
      wrap.append(label, input);
    } else {
      input = el('input');
      input.type = f.type === 'number' ? 'number' : 'text';
      if (f.placeholder) input.placeholder = f.placeholder;
      if (f.default) input.value = f.default;
      wrap.append(label, input);
    }
    input.id = id;
    input.dataset.flag = f.flag;
    input.dataset.kind = f.type;
    if (f.required) input.dataset.required = '1';
    input.addEventListener('input', renderPreview);
    input.addEventListener('change', renderPreview);
    box.appendChild(wrap);
  });

  let dl = $('runtags');
  if (!dl) { dl = el('datalist'); dl.id = 'runtags'; document.body.appendChild(dl); }
  dl.innerHTML = '';
  STATE.runs.filter((r) => r.kind === 'run').forEach((r) => {
    const o = el('option');
    o.value = r.tag;
    dl.appendChild(o);
  });
  renderPreview();
}

function collectArgs() {
  const args = [];
  $('fields').querySelectorAll('[data-flag]').forEach((input) => {
    const flag = input.dataset.flag;
    if (input.dataset.kind === 'flag') {
      if (input.checked) args.push(flag);
    } else if (input.value !== '' && input.value != null) {
      args.push(flag, input.value.trim());
    }
  });
  const extra = $('extra').value.trim();
  if (extra) args.push(...extra.split(/\s+/));
  return args;
}

function missingRequired() {
  const missing = [];
  $('fields').querySelectorAll('[data-required]').forEach((input) => {
    if (!input.value.trim()) missing.push(input.dataset.flag);
  });
  return missing;
}

function renderPreview() {
  $('cmdpreview').textContent =
    ['python cb.py', $('cmd').value, ...collectArgs()].join(' ');
}

// ── job running and the log ───────────────────────────────────────────────
let logCursor = 0;
let pollTimer = null;

function renderStatus() {
  const s = $('status');
  const job = STATE.job;
  if (!job) { s.className = 'status idle'; s.textContent = t('st.idle'); return; }
  if (job.running) {
    s.className = 'status busy';
    s.textContent = `${t('st.running')} · ${job.command} · ${job.elapsed}s`;
  } else if (job.code === 0) {
    s.className = 'status ok';
    s.textContent = `${t('st.done')} · ${job.command} · ${job.elapsed}s`;
  } else {
    s.className = 'status fail';
    s.textContent = `${t('st.failed')} · ${job.command} · exit ${job.code}`;
  }
  $('run').disabled = !!job.running;
  $('stop').disabled = !job.running;
}

function appendLog(lines) {
  const box = $('log');
  lines.forEach((line) => {
    const span = el('span', logClass(line), line + '\n');
    box.appendChild(span);
  });
  if ($('follow').checked) box.scrollTop = box.scrollHeight;
}

function logClass(line) {
  if (/\bERROR\b|Traceback|\bFAILED\b/.test(line)) return 'err';
  if (/\bWARNING\b/.test(line)) return 'warn';
  return '';
}

async function pollLog() {
  if (!STATE.job) return;
  try {
    const r = await api(`/api/log?job=${STATE.job.id}&from=${logCursor}`);
    if (r.lines.length) { appendLog(r.lines); logCursor = r.next; }
    STATE.job = r.job;
    $('logMeta').textContent = `${r.job.command} ${r.job.args.join(' ')}`;
    renderStatus();
    if (r.job.running) {
      pollTimer = setTimeout(pollLog, 700);
    } else {
      appendLog([`— exit ${r.job.code} after ${r.job.elapsed}s —`]);
      $('log').lastChild.className = r.job.code === 0 ? 'done' : 'err';
      refreshTrees();
      loadRuns();
    }
  } catch (e) {
    appendLog(['log poll failed: ' + e.message]);
  }
}

async function startRun() {
  const missing = missingRequired();
  if (missing.length) {
    appendLog([`${missing.join(', ')}${t('e.required')}`]);
    $('log').lastChild.className = 'err';
    return;
  }
  const command = $('cmd').value;
  const args = collectArgs();
  try {
    const r = await api('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ command, args }),
    });
    STATE.job = r.job;
    logCursor = 0;
    appendLog([`$ python cb.py ${command} ${args.join(' ')}`]);
    $('log').lastChild.className = 'meta';
    renderStatus();
    clearTimeout(pollTimer);
    pollLog();
  } catch (e) {
    appendLog([(e.body && e.body.detail) || e.message]);
    $('log').lastChild.className = 'err';
  }
}

async function stopRun() {
  if (!STATE.job) return;
  await api('/api/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ job: STATE.job.id }),
  }).catch(() => {});
  appendLog([`— ${t('r.stopped')} —`]);
  $('log').lastChild.className = 'meta';
}

// ── file trees ────────────────────────────────────────────────────────────
function makeTree(container, rootNames) {
  container.innerHTML = '';
  rootNames.forEach((root) => {
    const info = STATE.roots[root] || {};
    const block = el('div', 'root');
    const head = el('div', 'rootname' + (info.exists ? '' : ' missing'));
    const caret = el('span', 'caret', '▸');
    head.append(caret, el('span', null, root), el('span', 'rootpath', info.path || ''));
    const kids = el('div', 'children');
    kids.hidden = true;
    head.addEventListener('click', async () => {
      if (!info.exists) return;
      kids.hidden = !kids.hidden;
      caret.textContent = kids.hidden ? '▸' : '▾';
      if (!kids.dataset.loaded) {
        kids.dataset.loaded = '1';
        await fillDir(kids, root, '');
      }
    });
    block.append(head, kids);
    container.appendChild(block);
  });
}

async function fillDir(container, root, path) {
  container.innerHTML = '';
  let data;
  try {
    data = await api(`/api/ls?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`);
  } catch (e) {
    container.appendChild(el('div', 'note', e.message));
    return;
  }
  if (data.error) { container.appendChild(el('div', 'note', data.error)); return; }
  if (!data.dirs.length && !data.files.length) {
    container.appendChild(el('div', 'note', t('v.empty')));
    return;
  }
  data.dirs.forEach((d) => {
    const sub = path ? `${path}/${d.name}` : d.name;
    const node = el('div', 'node');
    const caret = el('span', 'caret', '▸');
    node.append(caret, el('span', 'nm', d.name));
    const kids = el('div', 'children');
    kids.hidden = true;
    node.addEventListener('click', async () => {
      kids.hidden = !kids.hidden;
      caret.textContent = kids.hidden ? '▸' : '▾';
      if (!kids.dataset.loaded) { kids.dataset.loaded = '1'; await fillDir(kids, root, sub); }
    });
    container.append(node, kids);
  });
  data.files.forEach((f) => {
    const sub = path ? `${path}/${f.name}` : f.name;
    const node = el('div', 'node');
    node.append(el('span', 'caret', ' '), el('span', 'nm', f.name),
                el('span', 'size', fmtSize(f.size)));
    node.addEventListener('click', () => {
      document.querySelectorAll('.node.selected').forEach((n) => n.classList.remove('selected'));
      node.classList.add('selected');
      openFile(root, sub);
    });
    container.appendChild(node);
  });
  if (data.truncated) container.appendChild(el('div', 'note', '…'));
}

function refreshTrees() {
  makeTree($('inputTree'), STATE.input_roots);
  makeTree($('outputTree'), STATE.output_roots);
}

// ── viewer ────────────────────────────────────────────────────────────────
function setViewer(name, actions, node) {
  $('viewerName').textContent = name;
  const act = $('viewerActions');
  act.innerHTML = '';
  (actions || []).forEach((a) => act.appendChild(a));
  const v = $('viewer');
  v.innerHTML = '';
  v.scrollTop = 0;
  v.appendChild(node);
}

async function openFile(root, path, offset = 0) {
  let data;
  try {
    data = await api(`/api/file?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}&offset=${offset}&limit=25`);
  } catch (e) {
    setViewer(path, [], el('div', 'empty', e.message));
    return;
  }
  VIEW.reload = () => openFile(root, path, offset);
  const label = `${root} / ${path}  ·  ${fmtSize(data.size)}`;

  if (data.kind === 'image') return viewImage(root, path, label);
  if (data.kind === 'records') return viewRecords(root, path, data, label);
  if (data.kind === 'run') return viewRun(data.json, label);
  if (data.kind === 'json') return viewJson(data, label);
  if (data.kind === 'text') return viewText(data, label);
  setViewer(label, [], el('div', 'empty', t('v.binary')));
}

function viewText(data, label) {
  const wrap = el('div');
  if (data.clipped) wrap.appendChild(el('div', 'hint', t('v.clipped')));
  wrap.appendChild(el('pre', 'textview', data.text));
  setViewer(label, [], wrap);
}

function viewImage(root, path, label) {
  const wrap = el('div', 'canvas-wrap');
  const canvas = el('canvas');
  const img = new Image();
  img.onload = () => {
    const maxW = Math.max(320, $('viewer').clientWidth - 24);
    const scale = Math.min(1, maxW / img.naturalWidth);
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    meta.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
  };
  img.onerror = () => { meta.textContent = 'could not decode'; };
  img.src = `/api/raw?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`;
  const meta = el('div', 'hint', '…');
  wrap.append(canvas, meta);

  const open = el('button', 'ghost small', t('v.open'));
  open.addEventListener('click', () => window.open(img.src, '_blank', 'noopener'));
  setViewer(label, [open], wrap);
}

function viewRecords(root, path, data, label) {
  const wrap = el('div');
  const pager = el('div', 'pager');
  const prev = el('button', 'ghost small', t('v.prev'));
  const next = el('button', 'ghost small', t('v.next'));
  prev.disabled = data.offset <= 0;
  next.disabled = data.offset + data.records.length >= data.total;
  prev.addEventListener('click', () => openFile(root, path, Math.max(0, data.offset - 25)));
  next.addEventListener('click', () => openFile(root, path, data.offset + 25));
  pager.append(prev, next, el('span', 'count',
    `${t('v.showing')} ${data.offset + 1}–${data.offset + data.records.length} ${t('v.of')} ${data.total} ${t('v.records')}`));
  wrap.appendChild(pager);

  data.records.forEach((rec, i) => {
    const d = el('details', 'record');
    const s = el('summary');
    s.appendChild(el('span', null, String(data.offset + i + 1).padStart(3, '0')));
    if (rec.id) s.appendChild(el('span', 'pill', rec.id));
    if (rec.eval_type) s.appendChild(el('span', 'pill', rec.eval_type));
    if (rec.track) s.appendChild(el('span', 'pill', rec.track));
    if (rec.split) s.appendChild(el('span', 'pill ' + (rec.split === 'train' ? 'warn' : ''), rec.split));
    const head = rec.question_ko || rec.question_en || rec.instruction || rec.text || '';
    s.appendChild(el('span', 'nm', String(head).slice(0, 90)));
    d.appendChild(s);
    const body = el('div', 'body');
    body.appendChild(kvList(rec));
    d.appendChild(body);
    wrap.appendChild(d);
  });
  setViewer(label, [], wrap);
}

function kvList(obj) {
  const dl = el('dl', 'kv');
  Object.entries(obj).forEach(([k, v]) => {
    dl.appendChild(el('dt', null, k));
    const text = (v && typeof v === 'object') ? JSON.stringify(v, null, 1) : String(v);
    dl.appendChild(el('dd', null, text.length > 4000 ? text.slice(0, 4000) + ' …' : text));
  });
  return dl;
}

function viewJson(data, label) {
  const wrap = el('div');
  if (data.truncated) wrap.appendChild(el('div', 'hint', t('v.truncated')));
  wrap.appendChild(jsonTree(data.json));
  setViewer(label, [], wrap);
}

function jsonTree(value, key) {
  if (value === null || typeof value !== 'object') {
    const span = el('div', 'jsontree');
    const cls = typeof value === 'number' ? 'n' : 's';
    if (key !== undefined) span.append(el('span', 'k', key + ': '));
    span.append(el('span', cls, JSON.stringify(value)));
    return span;
  }
  const d = el('details', 'jsontree');
  const isArr = Array.isArray(value);
  const n = isArr ? value.length : Object.keys(value).length;
  d.appendChild(el('summary', null,
    `${key !== undefined ? key + ' ' : ''}${isArr ? `[${n}]` : `{${n}}`}`));
  const entries = isArr ? value.map((v, i) => [String(i), v]) : Object.entries(value);
  entries.slice(0, 200).forEach(([k, v]) => d.appendChild(jsonTree(v, k)));
  if (entries.length > 200) d.appendChild(el('div', 'note', '…'));
  return d;
}

// ── run file → dashboard ──────────────────────────────────────────────────
function viewRun(run, label) {
  const summary = summariseRun(run);
  const wrap = el('div', 'dash');
  wrap.appendChild(runSection(summary));
  const raw = el('button', 'ghost small', t('v.raw'));
  raw.addEventListener('click', () => setViewer(label, [], jsonTree(run)));
  setViewer(label, [raw], wrap);
}

function summariseRun(run) {
  const metrics = [];
  Object.entries(run.tracks || {}).forEach(([track, body]) => {
    if (body.perplexity != null) {
      metrics.push({ track, type: 'chunks', metric: 'perplexity',
                     value: body.perplexity, n: body.items, lower_better: true });
    }
    Object.entries(body.by_type || {}).forEach(([kind, m]) => {
      ['correct', 'f1', 'key_f1', 'value_accuracy', 'abstained', 'ece', 'brier',
       'mean_inconsistency'].forEach((name) => {
        if (typeof m[name] === 'number') {
          metrics.push({ track, type: kind, metric: name, value: m[name], n: m.n,
                         ci95: m[name + '_ci95'], no_answer: m.no_answer,
                         lower_better: name === 'ece' || name === 'brier' });
        }
      });
    });
  });
  return { tag: run.tag, model: run.model, book: run.book, lang: run.lang,
           think: run.think, elapsed: run.elapsed_sec,
           headline: run.headline || {}, meta: run.meta || {}, metrics };
}

function runSection(r) {
  const box = el('div');
  const head = el('h3', null, `${r.tag || '(untagged)'} — ${r.model || ''}`);
  box.appendChild(head);
  const bits = [r.book, r.lang, r.think === null ? null : `think ${r.think}`,
                r.elapsed ? `${r.elapsed}s` : null].filter(Boolean);
  box.appendChild(el('p', 'note', bits.join(' · ')));

  if (Object.keys(r.headline).length) {
    const tw = el('div', 'tablewrap');
    const tbl = el('table', 'grid');
    const thead = el('thead');
    const hr = el('tr');
    hr.append(el('th', null, t('d.headline')), el('th', null, ''));
    thead.appendChild(hr); tbl.appendChild(thead);
    const tb = el('tbody');
    Object.entries(r.headline).forEach(([k, v]) => {
      const tr = el('tr');
      tr.append(el('td', null, k), el('td', 'num', fmtNum(v)));
      tb.appendChild(tr);
    });
    tbl.appendChild(tb); tw.appendChild(tbl); box.appendChild(tw);
  }

  const grid = el('div', 'metric-grid');
  r.metrics.forEach((m) => grid.appendChild(metricCard([{ tag: r.tag, ...m }], m)));
  box.appendChild(grid);
  return box;
}

function metricCard(series, meta) {
  const card = el('div', 'metric');
  const head = el('div', 'mhead');
  head.append(el('span', 'mname', `${meta.track}·${meta.type} ${meta.metric}`));
  if (meta.lower_better) head.append(el('span', 'pill warn', '↓ better'));
  if (meta.n) head.append(el('span', 'mn', `n=${meta.n}`));
  card.appendChild(head);

  const scale = meta.metric === 'perplexity'
    ? Math.max(...series.map((s) => s.value)) * 1.15 || 1 : 1;
  const bars = el('div', 'bars');
  series.forEach((s) => {
    const row = el('div', 'barrow');
    row.appendChild(el('span', 'tag', s.tag || ''));
    const track = el('div', 'bartrack');
    const fill = el('div', 'barfill' + (meta.lower_better ? ' lower' : ''));
    fill.style.width = Math.max(1, Math.min(100, (s.value / scale) * 100)) + '%';
    track.appendChild(fill);
    row.append(track, el('span', 'val', fmtNum(s.value)));
    row.title = s.ci95 ? `95% CI ${fmtNum(s.ci95[0])} – ${fmtNum(s.ci95[1])}` : '';
    bars.appendChild(row);
  });
  card.appendChild(bars);
  return card;
}

// ── runs list and the cross-run dashboard ─────────────────────────────────
const picked = new Set();

async function loadRuns() {
  try {
    const r = await api('/api/runs');
    STATE.runs = r.runs;
  } catch { STATE.runs = []; }
  renderRunList();
  renderFields();
}

function renderRunList() {
  const box = $('runList');
  box.innerHTML = '';
  if (!STATE.runs.length) { box.appendChild(el('div', 'note', t('d.none'))); return; }
  box.appendChild(el('div', 'note', t('d.select')));
  STATE.runs.forEach((r) => {
    const card = el('div', 'runcard');
    const rt = el('div', 'rt');
    rt.appendChild(el('span', 'rtag', r.kind === 'compare'
      ? `${r.base} → ${r.after}` : (r.tag || r.file)));
    if (r.book) rt.appendChild(el('span', 'pill', r.book));
    card.appendChild(rt);
    if (r.model) card.appendChild(el('div', 'rmodel', r.model));
    if (r.kind === 'run') {
      const lab = el('label', 'cmp');
      const cb = el('input');
      cb.type = 'checkbox';
      cb.checked = picked.has(r.file);
      cb.addEventListener('click', (ev) => {
        ev.stopPropagation();
        cb.checked ? picked.add(r.file) : picked.delete(r.file);
      });
      lab.append(cb, el('span', null, t('d.compare')));
      card.appendChild(lab);
    }
    card.addEventListener('click', () => {
      document.querySelectorAll('.runcard.selected').forEach((n) => n.classList.remove('selected'));
      card.classList.add('selected');
      if (r.kind === 'compare') showCompare(r);
      else openFile('out', 'runs/' + r.file);
    });
    box.appendChild(card);
  });
}

function showCompare(c) {
  const wrap = el('div', 'dash');
  wrap.appendChild(el('h3', null, `${c.base} → ${c.after}`));
  wrap.appendChild(el('p', 'note', t('d.cmpNote')));
  const tw = el('div', 'tablewrap');
  const tbl = el('table', 'grid');
  const thead = el('thead'); const hr = el('tr');
  ['metric', 'base', 'after', 'delta', 'test'].forEach((h) => hr.appendChild(el('th', null, h)));
  thead.appendChild(hr); tbl.appendChild(thead);
  const tb = el('tbody');
  Object.entries(c.headline || {}).forEach(([k, d]) => {
    const tr = el('tr');
    const sig = c.significance && c.significance[k];
    tr.append(el('td', null, k), el('td', 'num', fmtNum(d.base)),
              el('td', 'num', fmtNum(d.after)), el('td', 'num',
                (d.delta > 0 ? '+' : '') + fmtNum(d.delta)));
    const cell = el('td');
    if (sig) {
      const isSig = sig.significant === true || (sig.p_value != null && sig.p_value < 0.05);
      cell.appendChild(el('span', 'pill ' + (isSig ? 'good' : ''),
        sig.p_value != null ? `p=${sig.p_value}` : `[${fmtNum(sig.ci95_low)}, ${fmtNum(sig.ci95_high)}]`));
      cell.appendChild(el('span', 'pill ' + (isSig ? 'good' : 'warn'),
        isSig ? t('d.sig') : t('d.noise')));
    }
    tr.appendChild(cell);
    tb.appendChild(tr);
  });
  tbl.appendChild(tb); tw.appendChild(tbl); wrap.appendChild(tw);
  VIEW.reload = () => showCompare(c);
  setViewer(c.file, [], wrap);
}

function showDashboard() {
  const runs = STATE.runs.filter((r) => r.kind === 'run'
    && (picked.size === 0 || picked.has(r.file)));
  const wrap = el('div', 'dash');
  wrap.appendChild(el('h3', null, t('d.title')));
  wrap.appendChild(el('p', 'note', t('d.note')));
  if (!runs.length) {
    wrap.appendChild(el('div', 'empty', t('d.none')));
    VIEW.reload = showDashboard;
    setViewer(t('d.title'), [], wrap);
    return;
  }
  // group every metric across runs by track·type·metric so a bar chart per
  // metric compares the checkpoints on identical items -- the only comparison
  // this benchmark treats as meaningful.
  const groups = new Map();
  runs.forEach((r) => r.metrics.forEach((m) => {
    const key = `${m.track}|${m.type}|${m.metric}`;
    if (!groups.has(key)) groups.set(key, { meta: m, series: [] });
    groups.get(key).series.push({ tag: r.tag, ...m });
  }));
  const grid = el('div', 'metric-grid');
  [...groups.values()].forEach((g) => grid.appendChild(metricCard(g.series, g.meta)));
  wrap.appendChild(grid);

  const comparisons = STATE.runs.filter((r) => r.kind === 'compare');
  if (comparisons.length) {
    wrap.appendChild(el('h3', null, t('d.cmpTitle')));
    comparisons.forEach((c) => {
      const b = el('button', 'ghost small', c.file);
      b.addEventListener('click', () => showCompare(c));
      wrap.appendChild(b);
    });
  }
  VIEW.reload = showDashboard;
  setViewer(t('d.title'), [], wrap);
}

// ── settings ──────────────────────────────────────────────────────────────
async function loadConfig() {
  const c = await api('/api/config');
  $('configPath').textContent = c.path || '(built-in defaults)';
  $('configText').value = c.text || '';
  $('configText').disabled = !c.path;
  $('saveConfig').disabled = !c.path;
  const dl = $('roots');
  dl.innerHTML = '';
  Object.entries(STATE.roots).forEach(([k, v]) => {
    dl.appendChild(el('dt', null, k));
    dl.appendChild(el('dd', v.exists ? '' : 'missing', v.path));
  });
}

async function saveConfig() {
  const msg = $('configMsg');
  try {
    const r = await api('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: $('configText').value }),
    });
    msg.className = 'msg ok';
    msg.textContent = t('e.saved') + ' ' + r.note;
  } catch (e) {
    msg.className = 'msg err';
    msg.textContent = e.message;
  }
}

// ── splitters ─────────────────────────────────────────────────────────────
function initSplitters() {
  const saved = JSON.parse(localStorage.getItem('kc.split') || '{}');
  if (saved.left) $('left').style.width = saved.left + 'px';
  if (saved.right) $('right').style.width = saved.right + 'px';
  if (saved.log) $('logpanel').style.height = saved.log + 'px';

  document.querySelectorAll('.splitter').forEach((sp) => {
    sp.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      sp.setPointerCapture(ev.pointerId);
      sp.classList.add('dragging');
      const which = sp.dataset.split;
      const startX = ev.clientX, startY = ev.clientY;
      const leftW = $('left').offsetWidth;
      const rightW = $('right').offsetWidth;
      const logH = $('logpanel').offsetHeight;

      const move = (e) => {
        if (which === 'left') {
          $('left').style.width = Math.max(180, Math.min(620, leftW + (e.clientX - startX))) + 'px';
        } else if (which === 'right') {
          $('right').style.width = Math.max(180, Math.min(620, rightW - (e.clientX - startX))) + 'px';
        } else {
          $('logpanel').style.height = Math.max(60, Math.min(window.innerHeight - 220, logH - (e.clientY - startY))) + 'px';
        }
      };
      const up = () => {
        sp.classList.remove('dragging');
        sp.removeEventListener('pointermove', move);
        sp.removeEventListener('pointerup', up);
        localStorage.setItem('kc.split', JSON.stringify({
          left: $('left').offsetWidth, right: $('right').offsetWidth,
          log: $('logpanel').offsetHeight,
        }));
      };
      sp.addEventListener('pointermove', move);
      sp.addEventListener('pointerup', up);
    });
  });
}

// ── tabs ──────────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      const panel = tab.closest('.panel');
      panel.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
      panel.querySelectorAll('.tabpane').forEach((x) => x.classList.remove('active'));
      tab.classList.add('active');
      panel.querySelector('#pane-' + tab.dataset.tab).classList.add('active');
    });
  });
}

// ── boot ──────────────────────────────────────────────────────────────────
async function boot() {
  applyTheme(localStorage.getItem('kc.theme') || 'dark');
  initTabs();
  initSplitters();

  const s = await api('/api/state');
  STATE.commands = s.commands;
  STATE.roots = s.roots;
  STATE.input_roots = s.input_roots;
  STATE.output_roots = s.output_roots;
  STATE.job = s.job;

  renderCommands();
  refreshTrees();
  await loadConfig();
  await loadRuns();
  renderFields();
  applyLang();
  renderStatus();

  if (STATE.job && STATE.job.running) { logCursor = 0; pollLog(); }

  $('cmd').addEventListener('change', () => {
    localStorage.setItem('kc.cmd', $('cmd').value);
    renderFields();
  });
  $('extra').addEventListener('input', renderPreview);
  $('run').addEventListener('click', startRun);
  $('stop').addEventListener('click', stopRun);
  $('saveConfig').addEventListener('click', saveConfig);
  $('refreshRuns').addEventListener('click', loadRuns);
  $('dashboard').addEventListener('click', showDashboard);
  $('clearLog').addEventListener('click', () => { $('log').innerHTML = ''; });
  $('theme').addEventListener('click', () =>
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  $('lang').addEventListener('click', () => {
    LANG = LANG === 'ko' ? 'en' : 'ko';
    localStorage.setItem('kc.lang', LANG);
    applyLang();
  });
}

boot().catch((e) => {
  document.body.insertAdjacentHTML('afterbegin',
    `<pre style="padding:12px;color:#d4705f">boot failed: ${e.message}</pre>`);
});
