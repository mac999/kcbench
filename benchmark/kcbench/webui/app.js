/* kcbench console ---------------------------------------------------------
   Talks only to the Flask endpoints in webview.py. Every number it draws is
   read back out of a run file that an ordinary cb.py command wrote.

   The page is arranged the way the benchmark is used, not the way the disk is
   laid out: the frozen items on the left, the checkpoints scored against them
   on the right, and the comparison between checkpoints in the middle. */

'use strict';

// ── i18n ──────────────────────────────────────────────────────────────────
const I18N = {
  en: {
    'brand.sub': 'benchmark console',
    'run.start': 'Run', 'run.stop': 'Stop', 'run.extra': 'extra flags',
    'tab.tracks': 'Benchmark', 'tab.sources': 'Sources', 'tab.settings': 'Settings',
    'tab.runs': 'Runs', 'tab.files': 'Files',
    'tracks.hint': 'The frozen item sets a run is scored on — the answer key. Open one to read its items.',
    'sources.hint': 'What the build stages read: the corpus and the generated training data. Not needed for scoring.',
    'runs.hint': 'One card per scored checkpoint. Tick several and press Chart to compare them on identical items.',
    'files.hint': 'Everything the benchmark has written. Selecting a folder of run files charts it.',
    'cfg.save': 'Save',
    'cfg.hint': 'Saved to config.json, with the previous version kept as .bak. Takes effect on the next command; restart the webview to re-resolve its own paths.',
    'cfg.roots': 'Resolved paths',
    'runs.chart': 'Chart', 'runs.refresh': 'Refresh',
    'log.title': 'Log', 'log.follow': 'Follow', 'log.clear': 'Clear',
    'viewer.empty': 'Start here',
    'st.idle': 'idle', 'st.running': 'running', 'st.done': 'done', 'st.failed': 'failed',
    'v.records': 'records', 'v.showing': 'showing', 'v.of': 'of',
    'v.prev': 'Previous', 'v.next': 'Next', 'v.raw': 'Raw JSON',
    'v.open': 'Open full size', 'v.clipped': 'clipped for display',
    'v.truncated': 'summarised — the file is too large to send whole',
    'v.binary': 'Binary file. Nothing to show.', 'v.empty': 'This directory is empty.',
    'v.notfound': 'not found on this machine',
    'v.folderNone': 'No run files in this folder. Folders with run files draw as a chart.',
    'd.title': 'Scored checkpoints', 'd.folder': 'Runs in',
    'd.none': 'No run files yet. Score a model with eval, ppl, rag or ece and it appears here.',
    'd.note': 'Every bar is read from a run file. Metrics share a 0–1 axis; perplexity has its own and lower is better. Whiskers are 95% Wilson intervals where the run recorded them. Only checkpoints scored on identical items are drawn in the same panel.',
    'd.cmpTitle': 'Paired comparisons', 'd.cmpNote': 'Written by cb.py compare. McNemar for accuracies, paired bootstrap for F1. An interval straddling zero, or a p-value near 1, means the data cannot tell the change from noise.',
    'd.sig': 'significant', 'd.noise': 'noise', 'd.delta': 'Change, after − base',
    'd.headline': 'Headline', 'd.cmpRow': 'comparison', 'd.items': 'items',
    'w.title': 'A benchmark run, in order',
    'w.lead': 'This tool answers one question: did fine-tuning on the corpus teach the model anything? Each step below loads its command into the bar above.',
    'w.1t': 'Build the benchmark, once', 'w.1d': 'Split the corpus, mine the items, write the training split with held-out documents removed. Do this before any training.',
    'w.2t': 'Baseline the checkpoint you will fine-tune', 'w.2d': 'Score the untrained model closed book and open book, and on the probe. Every later number is meaningless without this "before".',
    'w.3t': 'Train, on data/train and nothing else', 'w.3d': 'Outside this tool: training/dapt.py then training/sft.py. Do not score while training on a unified-memory box.',
    'w.4t': 'Register the fine-tune exactly like its base', 'w.4d': 'Outside this tool: ollama create with the base model\'s TEMPLATE and stop tokens. A mismatch here reads as a model failure that is not one.',
    'w.5t': 'Score the fine-tune on the same items', 'w.5d': 'Same tracks, same flags, same config as the baseline. Only the model tag changes.',
    'w.6t': 'Compare. This is the answer', 'w.6d': 'Paired significance on identical items. Read the probe against the held-out track.',
    'w.7t': 'Optional: what a score cannot see', 'w.7d': 'ece asks whether the model\'s confidence is worth anything; selfcheck looks for hallucination without an answer key.',
    'w.load': 'Load', 'w.outside': 'outside this tool',
    'e.saved': 'Saved.', 'e.required': 'is required',
    'r.stopped': 'stop requested',
    't.contam': 'train-side', 't.fuzzy': 'fuzzy match', 't.items': 'items',
  },
  ko: {
    'brand.sub': '벤치마크 콘솔',
    'run.start': '실행', 'run.stop': '중지', 'run.extra': '추가 플래그',
    'tab.tracks': '벤치마크', 'tab.sources': '소스', 'tab.settings': '설정',
    'tab.runs': '실행 결과', 'tab.files': '파일',
    'tracks.hint': '채점의 기준이 되는 고정 문항 세트 — 정답지입니다. 열어서 문항을 읽으십시오.',
    'sources.hint': '빌드 단계가 읽는 것: 코퍼스와 생성된 학습 데이터. 채점에는 필요 없습니다.',
    'runs.hint': '채점된 체크포인트마다 카드 하나. 여러 개를 체크하고 차트를 누르면 동일 문항 위에서 비교합니다.',
    'files.hint': '벤치마크가 기록한 모든 파일. 실행 결과 파일이 든 폴더를 고르면 차트로 그립니다.',
    'cfg.save': '저장',
    'cfg.hint': 'config.json에 저장하고 이전 내용은 .bak으로 남깁니다. 다음 명령부터 적용되며, 웹뷰 자체의 경로를 다시 잡으려면 재시작하십시오.',
    'cfg.roots': '해석된 경로',
    'runs.chart': '차트', 'runs.refresh': '새로고침',
    'log.title': '로그', 'log.follow': '자동 스크롤', 'log.clear': '지우기',
    'viewer.empty': '시작하기',
    'st.idle': '대기', 'st.running': '실행 중', 'st.done': '완료', 'st.failed': '실패',
    'v.records': '레코드', 'v.showing': '표시', 'v.of': '/',
    'v.prev': '이전', 'v.next': '다음', 'v.raw': '원본 JSON',
    'v.open': '원본 크기로 열기', 'v.clipped': '표시용으로 잘림',
    'v.truncated': '파일이 너무 커서 구조만 요약했습니다',
    'v.binary': '바이너리 파일입니다. 표시할 내용이 없습니다.', 'v.empty': '빈 디렉터리입니다.',
    'v.notfound': '이 장비에 없음',
    'v.folderNone': '이 폴더에 실행 결과 파일이 없습니다. 결과 파일이 든 폴더는 차트로 그려집니다.',
    'd.title': '채점된 체크포인트', 'd.folder': '실행 결과 폴더',
    'd.none': '아직 실행 결과가 없습니다. eval·ppl·rag·ece로 모델을 채점하면 여기에 나타납니다.',
    'd.note': '모든 막대는 실행 결과 파일에서 읽은 값입니다. 지표는 0–1 축을 공유하고, perplexity는 자체 축이며 낮을수록 좋습니다. 수염은 실행이 기록한 95% Wilson 구간입니다. 같은 패널에는 동일 문항으로 채점된 체크포인트만 그립니다.',
    'd.cmpTitle': '짝지은 비교', 'd.cmpNote': 'cb.py compare가 쓴 값입니다. 정확도는 McNemar, F1은 paired bootstrap. 구간이 0을 걸치거나 p값이 1에 가까우면 변화를 잡음과 구별할 수 없다는 뜻입니다.',
    'd.sig': '유의미', 'd.noise': '잡음', 'd.delta': '변화량, 이후 − 기준',
    'd.headline': '대표 지표', 'd.cmpRow': '비교', 'd.items': '문항',
    'w.title': '벤치마크 한 바퀴, 순서대로',
    'w.lead': '이 도구가 답하는 질문은 하나입니다: 코퍼스로 파인튜닝한 것이 모델에 무언가를 가르쳤는가? 아래 각 단계를 누르면 명령이 상단 바에 채워집니다.',
    'w.1t': '벤치마크 빌드 — 한 번만', 'w.1d': '코퍼스를 나누고, 문항을 채굴하고, 홀드아웃 문서를 뺀 학습 분할을 씁니다. 어떤 학습보다도 먼저 하십시오.',
    'w.2t': '파인튜닝할 체크포인트의 기준값', 'w.2d': '학습 전 모델을 closed book·open book·probe로 채점합니다. 이 "이전" 값이 없으면 이후 숫자는 의미가 없습니다.',
    'w.3t': '학습 — data/train만으로', 'w.3d': '이 도구 밖: training/dapt.py 다음 training/sft.py. 통합 메모리 장비에서는 학습 중에 채점하지 마십시오.',
    'w.4t': '파인튜닝 모델을 베이스와 똑같이 등록', 'w.4d': '이 도구 밖: 베이스 모델의 TEMPLATE과 stop 토큰으로 ollama create. 여기서 어긋나면 모델 결함이 아닌 것이 모델 결함처럼 보입니다.',
    'w.5t': '파인튜닝 모델을 같은 문항으로 채점', 'w.5d': '기준값과 같은 트랙, 같은 플래그, 같은 설정. 모델 태그만 바뀝니다.',
    'w.6t': '비교 — 이것이 답입니다', 'w.6d': '동일 문항 위의 짝지은 유의성 검정. probe를 홀드아웃 트랙과 나란히 읽으십시오.',
    'w.7t': '선택: 점수가 보지 못하는 것', 'w.7d': 'ece는 모델의 확신이 믿을 만한지, selfcheck는 정답지 없이 환각 신호를 찾습니다.',
    'w.load': '불러오기', 'w.outside': '이 도구 밖',
    'e.saved': '저장했습니다.', 'e.required': ' 값이 필요합니다',
    'r.stopped': '중지 요청됨',
    't.contam': '학습 측', 't.fuzzy': '퍼지 매칭', 't.items': '문항',
  },
};

let LANG = localStorage.getItem('kc.lang') || 'en';
const t = (k) => (I18N[LANG] && I18N[LANG][k]) || I18N.en[k] || k;

function applyLang() {
  document.documentElement.lang = LANG;
  document.querySelectorAll('[data-i18n]').forEach((n) => { n.textContent = t(n.dataset.i18n); });
  document.querySelectorAll('[data-i18n-ph]').forEach((n) => { n.placeholder = t(n.dataset.i18nPh); });
  $('lang').textContent = LANG === 'ko' ? '한' : 'EN';
  renderStatus();
  if (STATE.commands) renderFields();
  renderTracks();
  renderRunList();
  refreshTrees();
  if (VIEW.reload) VIEW.reload();
}

// ── helpers ───────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
const fmtSize = (b) => (b == null ? '' : b < 1024 ? b + ' B' : b < 1048576 ? (b / 1024).toFixed(0) + ' K' : (b / 1048576).toFixed(1) + ' M');
const fmtNum = (v) => (typeof v === 'number' ? (Math.abs(v) >= 100 ? v.toFixed(2) : v.toFixed(3)) : String(v));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6'];

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
  if (!r.ok) throw Object.assign(new Error(body.error || `HTTP ${r.status}`), { body });
  return body;
}

const STATE = { commands: null, roots: {}, job: null, runs: [], tracks: [], picked: new Set() };
const VIEW = { reload: null };

// ── theme ─────────────────────────────────────────────────────────────────
function applyTheme(mode) {
  document.documentElement.dataset.theme = mode;
  localStorage.setItem('kc.theme', mode);
  $('theme').textContent = mode === 'dark' ? '◑' : '◐';
  if (VIEW.reload) VIEW.reload();   // canvases take their colours from the theme
}

// ── command bar ───────────────────────────────────────────────────────────
function renderCommands() {
  const sel = $('cmd');
  sel.innerHTML = '';
  const groups = {};
  Object.entries(STATE.commands).forEach(([name, spec]) => { (groups[spec.group] ||= []).push(name); });
  Object.entries(groups).forEach(([group, names]) => {
    const og = el('optgroup'); og.label = group;
    names.forEach((n) => { const o = el('option', null, n); o.value = n; og.appendChild(o); });
    sel.appendChild(og);
  });
  sel.value = localStorage.getItem('kc.cmd') || 'eval';
  if (!sel.value) sel.value = 'build';
}

function renderFields(preset) {
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
      input = el('input'); input.type = 'checkbox'; wrap.append(input, label);
    } else if (f.type === 'select') {
      input = el('select');
      (f.options || []).forEach((o) => { const opt = el('option', null, o || '—'); opt.value = o; input.appendChild(opt); });
      wrap.append(label, input);
    } else if (f.type === 'run') {
      input = el('input'); input.type = 'text'; input.setAttribute('list', 'runtags');
      wrap.classList.add('wide'); wrap.append(label, input);
    } else {
      input = el('input'); input.type = f.type === 'number' ? 'number' : 'text';
      if (f.placeholder) input.placeholder = f.placeholder;
      if (f.default) input.value = f.default;
      wrap.append(label, input);
    }
    input.id = id; input.dataset.flag = f.flag; input.dataset.kind = f.type;
    if (f.required) input.dataset.required = '1';
    if (preset && preset[f.flag] !== undefined) {
      if (f.type === 'flag') input.checked = !!preset[f.flag]; else input.value = preset[f.flag];
    }
    input.addEventListener('input', renderPreview);
    input.addEventListener('change', renderPreview);
    box.appendChild(wrap);
  });
  let dl = $('runtags');
  if (!dl) { dl = el('datalist'); dl.id = 'runtags'; document.body.appendChild(dl); }
  dl.innerHTML = '';
  STATE.runs.filter((r) => r.kind === 'run').forEach((r) => { const o = el('option'); o.value = r.tag; dl.appendChild(o); });
  renderPreview();
}

function loadCommand(name, preset) {
  $('cmd').value = name;
  localStorage.setItem('kc.cmd', name);
  renderFields(preset);
  $('cmd').focus();
}

function collectArgs() {
  const args = [];
  $('fields').querySelectorAll('[data-flag]').forEach((input) => {
    const flag = input.dataset.flag;
    if (input.dataset.kind === 'flag') { if (input.checked) args.push(flag); }
    else if (input.value !== '' && input.value != null) args.push(flag, input.value.trim());
  });
  const extra = $('extra').value.trim();
  if (extra) args.push(...extra.split(/\s+/));
  return args;
}
function missingRequired() {
  const m = [];
  $('fields').querySelectorAll('[data-required]').forEach((i) => { if (!i.value.trim()) m.push(i.dataset.flag); });
  return m;
}
function renderPreview() {
  $('cmdpreview').textContent = ['python cb.py', $('cmd').value, ...collectArgs()].join(' ');
}

// ── jobs and the log ──────────────────────────────────────────────────────
let logCursor = 0, pollTimer = null;

function renderStatus() {
  const s = $('status'), job = STATE.job;
  if (!job) { s.className = 'status idle'; s.textContent = t('st.idle'); }
  else if (job.running) { s.className = 'status busy'; s.textContent = `${t('st.running')} · ${job.command} · ${job.elapsed}s`; }
  else if (job.code === 0) { s.className = 'status ok'; s.textContent = `${t('st.done')} · ${job.command} · ${job.elapsed}s`; }
  else { s.className = 'status fail'; s.textContent = `${t('st.failed')} · ${job.command} · exit ${job.code}`; }
  $('run').disabled = !!(job && job.running);
  $('stop').disabled = !(job && job.running);
}
function logClass(line) {
  if (/\bERROR\b|Traceback|\bFAILED\b/.test(line)) return 'err';
  if (/\bWARNING\b/.test(line)) return 'warn';
  return '';
}
function appendLog(lines, cls) {
  const box = $('log');
  lines.forEach((line) => box.appendChild(el('span', cls || logClass(line), line + '\n')));
  if ($('follow').checked) box.scrollTop = box.scrollHeight;
}
async function pollLog() {
  if (!STATE.job) return;
  try {
    const r = await api(`/api/log?job=${STATE.job.id}&from=${logCursor}`);
    if (r.lines.length) { appendLog(r.lines); logCursor = r.next; }
    STATE.job = r.job;
    $('logMeta').textContent = `${r.job.command} ${r.job.args.join(' ')}`;
    renderStatus();
    if (r.job.running) pollTimer = setTimeout(pollLog, 700);
    else {
      appendLog([`— exit ${r.job.code} after ${r.job.elapsed}s —`], r.job.code === 0 ? 'done' : 'err');
      refreshTrees(); loadRuns(); loadTracks();
    }
  } catch (e) { appendLog(['log poll failed: ' + e.message], 'err'); }
}
async function startRun() {
  const missing = missingRequired();
  if (missing.length) { appendLog([`${missing.join(', ')}${t('e.required')}`], 'err'); return; }
  const command = $('cmd').value, args = collectArgs();
  try {
    const r = await api('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ command, args }) });
    STATE.job = r.job; logCursor = 0;
    appendLog([`$ python cb.py ${command} ${args.join(' ')}`], 'meta');
    renderStatus(); clearTimeout(pollTimer); pollLog();
  } catch (e) { appendLog([(e.body && e.body.detail) || e.message], 'err'); }
}
async function stopRun() {
  if (!STATE.job) return;
  await api('/api/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ job: STATE.job.id }) }).catch(() => {});
  appendLog([`— ${t('r.stopped')} —`], 'meta');
}

// ── left: the benchmark (tracks) ──────────────────────────────────────────
async function loadTracks() {
  try { STATE.tracks = (await api('/api/tracks')).tracks; } catch { STATE.tracks = []; }
  renderTracks();
}
function renderTracks() {
  const box = $('trackList');
  if (!box) return;
  box.innerHTML = '';
  STATE.tracks.forEach((tr) => {
    const card = el('div', 'trackcard');
    const head = el('div', 'tt');
    head.append(el('span', 'tname', tr.usecase || tr.track || tr.file.replace('.jsonl', '')),
                el('span', 'tn', `${tr.n} ${t('t.items')}`));
    card.appendChild(head);
    const meta = el('div', 'tmeta');
    Object.entries(tr.eval_types).forEach(([k, n]) => meta.appendChild(el('span', 'pill solid', `${k} ${n}`)));
    if (tr.contaminated) meta.appendChild(el('span', 'pill warn', `${t('t.contam')} ${tr.contaminated}`));
    if (tr.match_modes.fuzzy) meta.appendChild(el('span', 'pill', t('t.fuzzy')));
    card.appendChild(meta);
    card.appendChild(el('div', 'tfile', tr.file));
    card.addEventListener('click', () => {
      selectOnly('.trackcard', card);
      openFile('out', tr.file);
    });
    box.appendChild(card);
  });
}
function selectOnly(selector, node) {
  document.querySelectorAll(selector + '.selected').forEach((n) => n.classList.remove('selected'));
  node.classList.add('selected');
}

// ── file trees (Sources on the left, Files on the right) ──────────────────
function makeTree(container, rootNames) {
  container.innerHTML = '';
  rootNames.forEach((root) => {
    const info = STATE.roots[root] || {};
    const block = el('div', 'root');
    const head = el('div', 'rootname' + (info.exists ? '' : ' missing'));
    const caret = el('span', 'caret', info.exists ? '▸' : ' ');
    head.append(caret, el('span', null, root));
    if (!info.exists) head.appendChild(el('span', 'pill bad', t('v.notfound')));
    head.appendChild(el('span', 'rootpath', info.path || ''));
    const kids = el('div', 'children'); kids.hidden = true;
    head.addEventListener('click', async () => {
      if (!info.exists) return;
      kids.hidden = !kids.hidden; caret.textContent = kids.hidden ? '▸' : '▾';
      if (!kids.dataset.loaded) { kids.dataset.loaded = '1'; await fillDir(kids, root, ''); }
    });
    block.append(head, kids);
    container.appendChild(block);
  });
}
async function fillDir(container, root, path) {
  container.innerHTML = '';
  let data;
  try { data = await api(`/api/ls?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`); }
  catch (e) { container.appendChild(el('div', 'note', e.message)); return; }
  if (data.error) { container.appendChild(el('div', 'note', data.error)); return; }
  if (!data.dirs.length && !data.files.length) { container.appendChild(el('div', 'note', t('v.empty'))); return; }
  data.dirs.forEach((d) => {
    const sub = path ? `${path}/${d.name}` : d.name;
    const node = el('div', 'node dir');
    const caret = el('span', 'caret', '▸');
    node.append(caret, el('span', 'nm', d.name));
    const kids = el('div', 'children'); kids.hidden = true;
    node.addEventListener('click', async () => {
      kids.hidden = !kids.hidden; caret.textContent = kids.hidden ? '▸' : '▾';
      if (!kids.dataset.loaded) { kids.dataset.loaded = '1'; await fillDir(kids, root, sub); }
      selectOnly('.node', node);
      openFolder(root, sub);              // a folder of results draws itself
    });
    container.append(node, kids);
  });
  data.files.forEach((f) => {
    const sub = path ? `${path}/${f.name}` : f.name;
    const node = el('div', 'node');
    node.append(el('span', 'caret', ' '), el('span', 'nm', f.name), el('span', 'size', fmtSize(f.size)));
    node.addEventListener('click', () => { selectOnly('.node', node); openFile(root, sub); });
    container.appendChild(node);
  });
  if (data.truncated) container.appendChild(el('div', 'note', '…'));
}
function refreshTrees() {
  if (!STATE.input_roots) return;
  makeTree($('inputTree'), STATE.input_roots);
  makeTree($('outputTree'), STATE.output_roots);
}

// ── viewer ────────────────────────────────────────────────────────────────
function setViewer(name, actions, node) {
  $('viewerName').textContent = name;
  const act = $('viewerActions'); act.innerHTML = '';
  (actions || []).forEach((a) => act.appendChild(a));
  const v = $('viewer'); v.innerHTML = ''; v.scrollTop = 0; v.appendChild(node);
}

function showWorkflow() {
  const wrap = el('div', 'workflow');
  wrap.appendChild(el('h2', null, t('w.title')));
  wrap.appendChild(el('p', null, t('w.lead')));
  const steps = el('div', 'steps');
  const plan = [
    { n: 1, cmd: 'build', preset: { '--strict': true }, ex: 'python cb.py build --strict' },
    { n: 2, cmd: 'eval', preset: { '-m': 'qwen3:8b', '--tag': 'base-closed', '--tracks': 'sft', '--closed-book': true }, ex: 'eval -m qwen3:8b --tag base-closed --tracks sft --closed-book' },
    { n: 3, outside: true, ex: 'python ../training/dapt.py  →  sft.py  →  merge.py' },
    { n: 4, outside: true, ex: 'ollama show qwen3:8b --modelfile > Modelfile.ft  →  ollama create my-ft:v1' },
    { n: 5, cmd: 'eval', preset: { '-m': 'my-ft:v1', '--tag': 'ft-closed', '--tracks': 'sft', '--closed-book': true }, ex: 'eval -m my-ft:v1 --tag ft-closed --tracks sft --closed-book' },
    { n: 6, cmd: 'compare', preset: { '--base': 'base-closed', '--after': 'ft-closed' }, ex: 'compare --base base-closed --after ft-closed' },
    { n: 7, cmd: 'ece', preset: { '-m': 'my-ft:v1', '--tag': 'ft-ece', '--tracks': 'sft', '--closed-book': true }, ex: 'ece … / selfcheck …' },
  ];
  plan.forEach((p) => {
    const s = el('div', 'step' + (p.outside ? ' outside' : ''));
    s.appendChild(el('div', 'no', String(p.n)));
    const body = el('div');
    body.appendChild(el('div', 'st', t(`w.${p.n}t`)));
    body.appendChild(el('div', 'sd', t(`w.${p.n}d`)));
    body.appendChild(el('div', 'sx', p.ex));
    s.appendChild(body);
    if (p.outside) s.appendChild(el('span', 'pill', t('w.outside')));
    else {
      const b = el('button', 'small', t('w.load'));
      b.addEventListener('click', () => loadCommand(p.cmd, p.preset));
      s.appendChild(b);
    }
    steps.appendChild(s);
  });
  wrap.appendChild(steps);
  VIEW.reload = showWorkflow;
  setViewer(t('viewer.empty'), [], wrap);
}

async function openFile(root, path, offset = 0) {
  let data;
  try { data = await api(`/api/file?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}&offset=${offset}&limit=25`); }
  catch (e) { setViewer(path, [], el('div', 'note', e.message)); return; }
  VIEW.reload = () => openFile(root, path, offset);
  const label = `${root} / ${path}  ·  ${fmtSize(data.size)}`;
  if (data.kind === 'image') return viewImage(root, path, label);
  if (data.kind === 'records') return viewRecords(root, path, data, label);
  if (data.kind === 'run') return viewRun(data.json, label);
  if (data.kind === 'json') return viewJson(data, label);
  if (data.kind === 'text') return viewText(data, label);
  setViewer(label, [], el('div', 'note', t('v.binary')));
}

async function openFolder(root, path) {
  let data;
  try { data = await api(`/api/runs?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`); }
  catch (e) { return; }
  const label = `${root} / ${path}`;
  if (!data.runs.length) {
    VIEW.reload = () => openFolder(root, path);
    setViewer(label, [], el('div', 'note', t('v.folderNone')));
    return;
  }
  renderDashboard(data.runs, `${t('d.folder')} ${label}`, () => openFolder(root, path));
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
  const meta = el('div', 'hint', '…');
  const img = new Image();
  img.onload = () => {
    const maxW = Math.max(320, $('viewer').clientWidth - 28);
    const scale = Math.min(1, maxW / img.naturalWidth);
    canvas.width = Math.round(img.naturalWidth * scale);
    canvas.height = Math.round(img.naturalHeight * scale);
    canvas.getContext('2d').drawImage(img, 0, 0, canvas.width, canvas.height);
    meta.textContent = `${img.naturalWidth} × ${img.naturalHeight}`;
  };
  img.onerror = () => { meta.textContent = 'could not decode'; };
  img.src = `/api/raw?root=${encodeURIComponent(root)}&path=${encodeURIComponent(path)}`;
  wrap.append(canvas, meta);
  const open = el('button', 'ghost small', t('v.open'));
  open.addEventListener('click', () => window.open(img.src, '_blank', 'noopener'));
  setViewer(label, [open], wrap);
}

// what a row's answer key is, in one line, by answer type
function answerOf(rec) {
  switch (rec.eval_type) {
    case 'numeric': return [rec.answer_value, rec.answer_unit].filter((x) => x != null).join(' ');
    case 'nameset': return Array.isArray(rec.answer) ? `${rec.answer.length}: ${rec.answer.slice(0, 3).join(' · ')}${rec.answer.length > 3 ? ' …' : ''}` : String(rec.answer ?? '');
    case 'faithfulness': return rec.context_matches ? `answer ${rec.answer_value ?? ''}` : 'abstain';
    case 'mapping': return rec.answer && typeof rec.answer === 'object' ? Object.entries(rec.answer).map(([k, v]) => `${k}=${v}`).join(' ') : '';
    default: return rec.answer != null ? String(rec.answer) : '';
  }
}

function viewRecords(root, path, data, label) {
  const wrap = el('div');
  const pager = el('div', 'pager');
  const prev = el('button', 'ghost small', t('v.prev')), next = el('button', 'ghost small', t('v.next'));
  prev.disabled = data.offset <= 0;
  next.disabled = data.offset + data.records.length >= data.total;
  prev.addEventListener('click', () => openFile(root, path, Math.max(0, data.offset - 25)));
  next.addEventListener('click', () => openFile(root, path, data.offset + 25));
  pager.append(prev, next, el('span', 'count', `${t('v.showing')} ${data.offset + 1}–${data.offset + data.records.length} ${t('v.of')} ${data.total} ${t('v.records')}`));
  wrap.appendChild(pager);

  data.records.forEach((rec, i) => {
    const d = el('details', 'record');
    const s = el('summary');
    s.appendChild(el('span', 'idx', String(data.offset + i + 1).padStart(3, '0')));
    const pills = el('span', 'pills');
    if (rec.eval_type) pills.appendChild(el('span', 'pill solid', rec.eval_type));
    if (rec.split === 'train') pills.appendChild(el('span', 'pill warn', t('t.contam')));
    if (rec.context_matches === false) pills.appendChild(el('span', 'pill', 'swapped'));
    s.appendChild(pills);
    const q = rec.question_ko || rec.question_en || rec.instruction || rec.text || rec.id || '';
    s.appendChild(el('span', 'q', String(q)));
    s.appendChild(el('span', 'ans', answerOf(rec)));
    d.appendChild(s);
    const body = el('div', 'body');
    body.appendChild(kvList(rec, ['answer', 'answer_value', 'answer_unit', 'answer_en']));
    d.appendChild(body);
    wrap.appendChild(d);
  });
  setViewer(label, [], wrap);
}
function kvList(obj, highlight) {
  const dl = el('dl', 'kv');
  const hi = new Set(highlight || []);
  Object.entries(obj).forEach(([k, v]) => {
    dl.appendChild(el('dt', null, k));
    const text = (v && typeof v === 'object') ? JSON.stringify(v, null, 1) : String(v);
    dl.appendChild(el('dd', hi.has(k) ? 'hi' : '', text.length > 4000 ? text.slice(0, 4000) + ' …' : text));
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
    if (key !== undefined) span.append(el('span', 'k', key + ': '));
    span.append(el('span', typeof value === 'number' ? 'n' : 's', JSON.stringify(value)));
    return span;
  }
  const d = el('details', 'jsontree');
  const isArr = Array.isArray(value), n = isArr ? value.length : Object.keys(value).length;
  d.appendChild(el('summary', null, `${key !== undefined ? key + ' ' : ''}${isArr ? `[${n}]` : `{${n}}`}`));
  const entries = isArr ? value.map((v, i) => [String(i), v]) : Object.entries(value);
  entries.slice(0, 200).forEach(([k, v]) => d.appendChild(jsonTree(v, k)));
  if (entries.length > 200) d.appendChild(el('div', 'note', '…'));
  return d;
}

// ── run files → charts ────────────────────────────────────────────────────
function summariseRun(run) {
  const metrics = [];
  Object.entries(run.tracks || {}).forEach(([track, body]) => {
    if (body.perplexity != null) metrics.push({ track, type: 'chunks', metric: 'perplexity', value: body.perplexity, n: body.items, lower_better: true });
    Object.entries(body.by_type || {}).forEach(([kind, m]) => {
      ['correct', 'f1', 'key_f1', 'value_accuracy', 'abstained', 'ece', 'brier', 'mean_inconsistency'].forEach((name) => {
        if (typeof m[name] === 'number') metrics.push({ track, type: kind, metric: name, value: m[name], n: m.n, ci95: m[name + '_ci95'], no_answer: m.no_answer, lower_better: name === 'ece' || name === 'brier' });
      });
    });
  });
  return { kind: 'run', tag: run.tag, model: run.model, book: run.book, lang: run.lang, think: run.think, elapsed: run.elapsed_sec, headline: run.headline || {}, meta: run.meta || {}, metrics };
}

function viewRun(run, label) {
  const r = summariseRun(run);
  const raw = el('button', 'ghost small', t('v.raw'));
  raw.addEventListener('click', () => setViewer(label, [], jsonTree(run)));
  const wrap = el('div', 'dash');
  wrap.appendChild(el('h3', null, `${r.tag || '(untagged)'} — ${r.model || ''}`));
  wrap.appendChild(el('p', 'note', [r.book, r.lang, r.think == null ? null : `think ${r.think}`, r.elapsed ? `${r.elapsed}s` : null].filter(Boolean).join(' · ')));
  if (Object.keys(r.headline).length) wrap.appendChild(headlineTable(r.headline));
  wrap.appendChild(metricGrid([r]));
  VIEW.reload = () => viewRun(run, label);
  setViewer(label, [raw], wrap);
}

function headlineTable(headline) {
  const tw = el('div', 'tablewrap'), tbl = el('table', 'grid');
  const thead = el('thead'), hr = el('tr');
  hr.append(el('th', null, t('d.headline')), el('th', null, ''));
  thead.appendChild(hr); tbl.appendChild(thead);
  const tb = el('tbody');
  Object.entries(headline).forEach(([k, v]) => { const tr = el('tr'); tr.append(el('td', null, k), el('td', 'num', fmtNum(v))); tb.appendChild(tr); });
  tbl.appendChild(tb); tw.appendChild(tbl);
  return tw;
}

// group every metric across runs by track·type·metric so a panel only ever
// compares checkpoints scored on identical items
function metricGrid(runs) {
  const groups = new Map();
  runs.forEach((r, ri) => r.metrics.forEach((m) => {
    const key = `${m.track}|${m.type}|${m.metric}`;
    if (!groups.has(key)) groups.set(key, { meta: m, series: [] });
    groups.get(key).series.push({ tag: r.tag, colour: ri, ...m });
  }));
  const grid = el('div', 'metric-grid');
  [...groups.values()].forEach((g) => grid.appendChild(metricCard(g.series, g.meta)));
  return grid;
}

function metricCard(series, meta) {
  const card = el('div', 'metric');
  const head = el('div', 'mhead');
  head.append(el('span', 'mname', `${meta.track} · ${meta.type} · ${meta.metric}`));
  if (meta.lower_better) head.append(el('span', 'pill warn', '↓ better'));
  if (meta.n) head.append(el('span', 'mn', `n=${meta.n}`));
  card.appendChild(head);
  const canvas = el('canvas');
  card.appendChild(canvas);
  requestAnimationFrame(() => drawBars(canvas, series, meta));
  return card;
}

// ── canvas charts ─────────────────────────────────────────────────────────
function prepCanvas(canvas, cssH) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.parentElement.clientWidth - 2 || 300;
  canvas.style.height = cssH + 'px';
  canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.font = '11px ' + cssVar('--mono');
  return { ctx, W: cssW, H: cssH };
}

function drawBars(canvas, series, meta) {
  const rowH = 22, padT = 6, padB = 18, labelW = 96, valW = 54;
  const { ctx, W, H } = prepCanvas(canvas, padT + rowH * series.length + padB);
  const x0 = labelW, x1 = W - valW, y0 = padT;
  const maxV = meta.metric === 'perplexity' ? Math.max(...series.map((s) => s.value)) * 1.15 || 1 : 1;
  const sx = (v) => x0 + (Math.max(0, Math.min(maxV, v)) / maxV) * (x1 - x0);

  const rule = cssVar('--rule'), faint = cssVar('--ink-faint'), ink = cssVar('--ink');
  // grid at 0 / .25 / .5 / .75 / 1 (or 4 even ticks on the perplexity axis)
  ctx.strokeStyle = rule; ctx.fillStyle = faint; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let i = 0; i <= 4; i++) {
    const v = (maxV * i) / 4, x = sx(v);
    ctx.beginPath(); ctx.moveTo(x, y0); ctx.lineTo(x, H - padB + 2); ctx.stroke();
    ctx.fillText(meta.metric === 'perplexity' ? v.toFixed(1) : v.toFixed(2), x, H - padB + 5);
  }
  series.forEach((s, i) => {
    const y = y0 + i * rowH + 4, h = rowH - 8;
    ctx.fillStyle = cssVar(SERIES[s.colour % SERIES.length]);
    ctx.fillRect(x0, y, Math.max(1, sx(s.value) - x0), h);
    if (s.ci95) {                       // 95% interval as a whisker
      ctx.strokeStyle = ink; ctx.lineWidth = 1;
      const cy = y + h / 2, a = sx(s.ci95[0]), b = sx(s.ci95[1]);
      ctx.beginPath(); ctx.moveTo(a, cy); ctx.lineTo(b, cy);
      ctx.moveTo(a, cy - 4); ctx.lineTo(a, cy + 4); ctx.moveTo(b, cy - 4); ctx.lineTo(b, cy + 4); ctx.stroke();
    }
    ctx.fillStyle = ink; ctx.textBaseline = 'middle';
    ctx.textAlign = 'right'; ctx.fillText(trunc(s.tag || '', 14), x0 - 6, y + h / 2);
    ctx.textAlign = 'left'; ctx.fillText(fmtNum(s.value), x1 + 6, y + h / 2);
  });
}

function drawDeltas(canvas, rows) {
  // rows: [{label, delta, lo, hi, significant}] -- a bar from zero, whiskers
  // for the bootstrap interval, coloured by whether the change survived its test
  const rowH = 24, padT = 6, padB = 18, labelW = 190, valW = 60;
  const { ctx, W, H } = prepCanvas(canvas, padT + rowH * rows.length + padB);
  const x0 = labelW, x1 = W - valW;
  const span = Math.max(0.05, ...rows.map((r) => Math.max(Math.abs(r.delta), Math.abs(r.lo ?? 0), Math.abs(r.hi ?? 0)))) * 1.15;
  const sx = (v) => x0 + ((v + span) / (2 * span)) * (x1 - x0);
  const rule = cssVar('--rule'), faint = cssVar('--ink-faint'), ink = cssVar('--ink');
  ctx.strokeStyle = rule; ctx.fillStyle = faint; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  [-span, -span / 2, 0, span / 2, span].forEach((v) => {
    const x = sx(v);
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, H - padB + 2); ctx.stroke();
    ctx.fillText((v > 0 ? '+' : '') + v.toFixed(2), x, H - padB + 5);
  });
  ctx.strokeStyle = ink; ctx.beginPath(); ctx.moveTo(sx(0), padT); ctx.lineTo(sx(0), H - padB + 2); ctx.stroke();
  rows.forEach((r, i) => {
    const y = padT + i * rowH + 5, h = rowH - 10, cy = y + h / 2;
    ctx.fillStyle = r.significant ? (r.delta >= 0 ? cssVar('--good') : cssVar('--bad')) : cssVar('--ink-faint');
    const a = sx(Math.min(0, r.delta)), b = sx(Math.max(0, r.delta));
    ctx.fillRect(a, y, Math.max(1, b - a), h);
    if (r.lo != null && r.hi != null) {
      ctx.strokeStyle = ink;
      const l = sx(r.lo), hh = sx(r.hi);
      ctx.beginPath(); ctx.moveTo(l, cy); ctx.lineTo(hh, cy);
      ctx.moveTo(l, cy - 4); ctx.lineTo(l, cy + 4); ctx.moveTo(hh, cy - 4); ctx.lineTo(hh, cy + 4); ctx.stroke();
    }
    ctx.fillStyle = ink; ctx.textBaseline = 'middle';
    ctx.textAlign = 'right'; ctx.fillText(trunc(r.label, 30), x0 - 6, cy);
    ctx.textAlign = 'left'; ctx.fillText((r.delta > 0 ? '+' : '') + fmtNum(r.delta), x1 + 6, cy);
  });
}
const trunc = (s, n) => (s.length > n ? s.slice(0, n - 1) + '…' : s);

// ── right: runs list, and the cross-run dashboard ─────────────────────────
async function loadRuns() {
  try { STATE.runs = (await api('/api/runs')).runs; } catch { STATE.runs = []; }
  renderRunList();
  if (STATE.commands) renderFields();
}

function renderRunList() {
  const box = $('runList');
  if (!box) return;
  box.innerHTML = '';
  const runs = STATE.runs.filter((r) => r.kind === 'run'), cmps = STATE.runs.filter((r) => r.kind === 'compare');
  if (!runs.length && !cmps.length) { box.appendChild(el('div', 'note', t('d.none'))); return; }
  runs.forEach((r, i) => {
    const card = el('div', 'runcard');
    const tag = el('div', 'rtag');
    const sw = el('span', 'sw'); sw.style.background = cssVar(SERIES[i % SERIES.length]);
    tag.append(sw, document.createTextNode(r.tag || r.file));
    card.appendChild(tag);
    const cb = el('input'); cb.type = 'checkbox'; cb.checked = STATE.picked.has(r.file);
    cb.addEventListener('click', (ev) => { ev.stopPropagation(); cb.checked ? STATE.picked.add(r.file) : STATE.picked.delete(r.file); });
    card.appendChild(cb);
    if (r.model) card.appendChild(el('div', 'rmodel', r.model));
    const pills = el('div', 'rpills');
    if (r.book) pills.appendChild(el('span', 'pill solid', r.book));
    if (r.think != null) pills.appendChild(el('span', 'pill', `think ${r.think}`));
    Object.entries(r.headline || {}).slice(0, 2).forEach(([k, v]) => pills.appendChild(el('span', 'pill', `${k.replace(/^track\d_|_score$/g, '')} ${fmtNum(v)}`)));
    card.appendChild(pills);
    card.addEventListener('click', () => { selectOnly('.runcard', card); openFile('out', 'runs/' + r.file); });
    box.appendChild(card);
  });
  if (cmps.length) {
    box.appendChild(el('div', 'rungroup', t('d.cmpTitle')));
    cmps.forEach((c) => {
      const card = el('div', 'runcard');
      card.appendChild(el('div', 'rtag', `${c.base} → ${c.after}`));
      card.appendChild(el('div', 'rmodel', c.file));
      card.addEventListener('click', () => { selectOnly('.runcard', card); showCompare(c); });
      box.appendChild(card);
    });
  }
}

function showDashboard() {
  const runs = STATE.runs.filter((r) => r.kind === 'run' && (STATE.picked.size === 0 || STATE.picked.has(r.file)));
  renderDashboard(runs, t('d.title'), showDashboard, STATE.runs.filter((r) => r.kind === 'compare'));
}

function renderDashboard(runs, title, reload, comparisons) {
  const wrap = el('div', 'dash');
  wrap.appendChild(el('h3', null, title));
  wrap.appendChild(el('p', 'note', t('d.note')));
  const plain = runs.filter((r) => r.kind === 'run');
  if (!plain.length) wrap.appendChild(el('div', 'note', t('d.none')));
  else {
    const legend = el('div', 'legend');
    plain.forEach((r, i) => {
      const item = el('span');
      const sw = el('span', 'sw'); sw.style.background = cssVar(SERIES[i % SERIES.length]);
      item.append(sw, document.createTextNode(`${r.tag}  ${r.model || ''} ${r.book ? '· ' + r.book : ''}`));
      legend.appendChild(item);
    });
    wrap.appendChild(legend);
    wrap.appendChild(metricGrid(plain));
  }
  const cmps = comparisons || runs.filter((r) => r.kind === 'compare');
  if (cmps.length) {
    wrap.appendChild(el('h3', null, t('d.cmpTitle')));
    wrap.appendChild(el('p', 'note', t('d.cmpNote')));
    cmps.forEach((c) => wrap.appendChild(compareBlock(c)));
  }
  VIEW.reload = reload;
  setViewer(title, [], wrap);
}

function compareRows(c) {
  return Object.entries(c.headline || {}).map(([k, d]) => {
    const sig = (c.significance || {})[k] || {};
    const significant = sig.significant === true || (sig.p_value != null && sig.p_value < 0.05);
    return { label: k, base: d.base, after: d.after, delta: d.delta, lo: sig.ci95_low, hi: sig.ci95_high, p: sig.p_value, test: sig.test, significant, tested: !!sig.test };
  });
}

function compareBlock(c) {
  const rows = compareRows(c);
  const block = el('div', 'metric wide');
  block.appendChild(el('div', 'mhead', `${c.base} → ${c.after}   ·   ${t('d.delta')}`));
  const canvas = el('canvas');
  block.appendChild(canvas);
  requestAnimationFrame(() => drawDeltas(canvas, rows));
  const tw = el('div', 'tablewrap'), tbl = el('table', 'grid');
  const thead = el('thead'), hr = el('tr');
  ['metric', 'base', 'after', 'Δ', 'test'].forEach((h) => hr.appendChild(el('th', null, h)));
  thead.appendChild(hr); tbl.appendChild(thead);
  const tb = el('tbody');
  rows.forEach((r) => {
    const tr = el('tr');
    tr.append(el('td', null, r.label), el('td', 'num', fmtNum(r.base)), el('td', 'num', fmtNum(r.after)), el('td', 'num', (r.delta > 0 ? '+' : '') + fmtNum(r.delta)));
    const cell = el('td');
    if (r.tested) {
      cell.appendChild(el('span', 'pill', r.p != null ? `p=${r.p}` : `[${fmtNum(r.lo)}, ${fmtNum(r.hi)}]`));
      cell.appendChild(el('span', 'pill ' + (r.significant ? 'good' : 'warn'), r.significant ? t('d.sig') : t('d.noise')));
    }
    tr.appendChild(cell); tb.appendChild(tr);
  });
  tbl.appendChild(tb); tw.appendChild(tbl); block.appendChild(tw);
  return block;
}

function showCompare(c) {
  const wrap = el('div', 'dash');
  wrap.appendChild(el('h3', null, `${c.base} → ${c.after}`));
  wrap.appendChild(el('p', 'note', t('d.cmpNote')));
  wrap.appendChild(compareBlock(c));
  VIEW.reload = () => showCompare(c);
  setViewer(c.file, [], wrap);
}

// ── settings ──────────────────────────────────────────────────────────────
async function loadConfig() {
  const c = await api('/api/config');
  $('configPath').textContent = c.path || '(built-in defaults)';
  $('configText').value = c.text || '';
  $('configText').disabled = !c.path; $('saveConfig').disabled = !c.path;
  const dl = $('roots'); dl.innerHTML = '';
  Object.entries(STATE.roots).forEach(([k, v]) => { dl.appendChild(el('dt', null, k)); dl.appendChild(el('dd', v.exists ? '' : 'missing', v.path)); });
}
async function saveConfig() {
  const msg = $('configMsg');
  try {
    const r = await api('/api/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: $('configText').value }) });
    msg.className = 'msg ok'; msg.textContent = t('e.saved') + ' ' + r.note;
  } catch (e) { msg.className = 'msg err'; msg.textContent = e.message; }
}

// ── splitters and tabs ────────────────────────────────────────────────────
function initSplitters() {
  const saved = JSON.parse(localStorage.getItem('kc.split') || '{}');
  if (saved.left) $('left').style.width = saved.left + 'px';
  if (saved.right) $('right').style.width = saved.right + 'px';
  if (saved.log) $('logpanel').style.height = saved.log + 'px';
  document.querySelectorAll('.splitter').forEach((sp) => {
    sp.addEventListener('pointerdown', (ev) => {
      ev.preventDefault(); sp.setPointerCapture(ev.pointerId); sp.classList.add('dragging');
      const which = sp.dataset.split, sx0 = ev.clientX, sy0 = ev.clientY;
      const lw = $('left').offsetWidth, rw = $('right').offsetWidth, lh = $('logpanel').offsetHeight;
      const move = (e) => {
        if (which === 'left') $('left').style.width = Math.max(200, Math.min(640, lw + (e.clientX - sx0))) + 'px';
        else if (which === 'right') $('right').style.width = Math.max(200, Math.min(640, rw - (e.clientX - sx0))) + 'px';
        else $('logpanel').style.height = Math.max(60, Math.min(window.innerHeight - 220, lh - (e.clientY - sy0))) + 'px';
      };
      const up = () => {
        sp.classList.remove('dragging'); sp.removeEventListener('pointermove', move); sp.removeEventListener('pointerup', up);
        localStorage.setItem('kc.split', JSON.stringify({ left: $('left').offsetWidth, right: $('right').offsetWidth, log: $('logpanel').offsetHeight }));
        if (VIEW.reload) VIEW.reload();   // canvases size to their column
      };
      sp.addEventListener('pointermove', move); sp.addEventListener('pointerup', up);
    });
  });
}
function initTabs() {
  document.querySelectorAll('.tab').forEach((tab) => tab.addEventListener('click', () => {
    const panel = tab.closest('.panel');
    panel.querySelectorAll('.tab').forEach((x) => x.classList.remove('active'));
    panel.querySelectorAll('.tabpane').forEach((x) => x.classList.remove('active'));
    tab.classList.add('active');
    panel.querySelector('#pane-' + tab.dataset.tab).classList.add('active');
  }));
}

// ── boot ──────────────────────────────────────────────────────────────────
async function boot() {
  applyTheme(localStorage.getItem('kc.theme') || 'dark');
  initTabs(); initSplitters();
  const s = await api('/api/state');
  Object.assign(STATE, { commands: s.commands, roots: s.roots, input_roots: s.input_roots, output_roots: s.output_roots, job: s.job });
  renderCommands();
  await Promise.all([loadConfig(), loadRuns(), loadTracks()]);
  refreshTrees();
  renderFields();
  applyLang();
  showWorkflow();
  if (STATE.job && STATE.job.running) { logCursor = 0; pollLog(); }

  $('cmd').addEventListener('change', () => { localStorage.setItem('kc.cmd', $('cmd').value); renderFields(); });
  $('extra').addEventListener('input', renderPreview);
  $('run').addEventListener('click', startRun);
  $('stop').addEventListener('click', stopRun);
  $('saveConfig').addEventListener('click', saveConfig);
  $('refreshRuns').addEventListener('click', () => { loadRuns(); loadTracks(); });
  $('dashboard').addEventListener('click', showDashboard);
  $('clearLog').addEventListener('click', () => { $('log').innerHTML = ''; });
  $('theme').addEventListener('click', () => applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));
  $('lang').addEventListener('click', () => { LANG = LANG === 'ko' ? 'en' : 'ko'; localStorage.setItem('kc.lang', LANG); applyLang(); });
  window.addEventListener('resize', () => { clearTimeout(window.__rs); window.__rs = setTimeout(() => VIEW.reload && VIEW.reload(), 150); });
}

boot().catch((e) => {
  document.body.insertAdjacentHTML('afterbegin', `<pre style="padding:12px;color:#d4705f">boot failed: ${e.message}</pre>`);
});
