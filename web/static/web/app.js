const $ = s => document.querySelector(s), $$ = s => [...document.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ico = n => `<svg class="i"><use href="#i-${n}"/></svg>`;
const store = { get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } }, set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} } };
const toast = (m, err) => { const t = document.createElement('div'); t.className = 'toast' + (err ? ' err' : ''); t.textContent = m; $('#toasts').append(t); setTimeout(() => t.remove(), 4200); };
const api = async (u, b) => { const r = await fetch(u, b ? {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify(b)} : {});
  const j = await r.json().catch(() => ({})); if (!r.ok) throw new Error(j.detail || r.statusText); return j; };
const money = n => '$' + Number(n).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
const roleBadge = r => `<span class="badge ${r === 'cfo' ? 'b-bad' : r === 'finance_manager' ? 'b-warn' : 'b-ok'}">${r === 'cfo' ? 'CFO' : r === 'finance_manager' ? 'Finance manager' : 'Clerk'}</span>`;
const tbl = (head, rows, num = []) => `<table><tr>${head.map((h, i) => `<th class="${num.includes(i) ? 'n' : ''}">${h}</th>`).join('')}</tr>${rows.map(r => `<tr>${r.map((c, i) => `<td class="${num.includes(i) ? 'n' : ''}">${c}</td>`).join('')}</tr>`).join('')}</table>`;

let cur = 'assistant';
// ---------- shell
const TITLES = {assistant:['Assistant','Grounded answers from your policies and contracts'], agent:['Maya · Payment agent','Chat with your finance-ops colleague — payments always need your approval'],
  extract:['Invoice extraction','Free text to validated JSON'], ledger:['Ledger','Live view of the finance database'],
  lab:['Model lab','Inspect the tokenizer and sample from our own GPT'], eval:['Evaluation','Quality, training and optimisation results']};
function go(v) {
  $$('.nav').forEach(b => b.classList.toggle('on', b.dataset.v === v)); $$('.view').forEach(s => s.classList.toggle('on', s.id === 'v-' + v));
  $('#ttl').textContent = TITLES[v][0]; $('#sub').textContent = TITLES[v][1]; cur = v; $('#newChat').style.display = (v === 'assistant' || v === 'agent') ? '' : 'none'; if (v === 'agent') agentInit();
  $('#side').classList.remove('open'); if (v === 'ledger') loadLedger(); if (v === 'eval') loadEval(); store.set('finmind.view', v);
}
$$('.nav').forEach(b => b.onclick = () => go(b.dataset.v)); $('#burger').onclick = () => $('#side').classList.toggle('open');
function theme(t) { document.documentElement.dataset.theme = t; $('#themeBtn').innerHTML = `${ico(t === 'dark' ? 'sun' : 'moon')}<span>${t === 'dark' ? 'Light mode' : 'Dark mode'}</span>`; store.set('finmind.theme', t); }
$('#themeBtn').onclick = () => theme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
theme(store.get('finmind.theme', matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light'));
api('/api/health').then(h => { $('#hdot').className = 'dot ok'; $('#htxt').textContent = `${h.providers.join(' → ')} · ${h.chunks} chunks`; }).catch(() => { $('#hdot').className = 'dot bad'; $('#htxt').textContent = 'API offline'; });

// ---------- assistant
const SUGGEST = ['Who must approve an invoice of $7,500?', 'What notice is needed to terminate the Acme Cloud agreement?', 'On which days are payments processed?', "What is the CEO's salary?"];
let chat = store.get('finmind.chat', []);
function bubble(m) {
  if (m.role === 'user') return `<div class="msg user"><div class="av u">You</div><div class="bub"><div class="txt">${esc(m.text)}</div></div></div>`;
  const cites = (m.cites || []).length ? `<div class="cites"><div class="mute" style="font-size:12px;font-weight:600">SOURCES</div>${m.cites.map(c => `<div class="cite"><span class="num">${c.n}</span><div>${esc(c.text)}<small>${ico('file').replace('class="i"', 'class="i" style="width:12px;height:12px;vertical-align:-2px"')} ${esc(c.doc)}</small></div></div>`).join('')}</div>` : '';
  const chatty = m.kind === 'chat';
  const status = chatty ? '' : m.declined ? '<span class="badge b-warn">Declined · not in documents</span>' : `<span class="badge b-ok">${ico('shield').replace('class="i"', 'class="i" style="width:13px;height:13px"')} Grounded · ${(m.cites || []).length} sources</span>`;
  const meta = m.pending ? '' : `<div class="meta">${status}
    ${chatty ? '' : `<span class="badge b-mute">${esc(m.mode || '')}${m.rerank ? ' + rerank' : ''}</span>`}<span class="badge b-mute">${m.ms} ms</span>
    <button class="btn sm ghost" data-copy="${esc(m.text)}" style="margin-left:auto">${ico('copy').replace('class="i"', 'class="i" style="width:14px;height:14px"')}Copy</button></div>`;
  return `<div class="msg bot"><div class="av a"><svg class="i" style="width:17px;height:17px;stroke-width:2"><use href="#i-spark"/></svg></div><div class="bub"><div class="txt">${m.pending && !m.text ? '<span class="typing"><span></span><span></span><span></span></span>' : esc(m.text)}</div>${cites}${meta}</div></div>`;
}
function renderChat() {
  const t = $('#thread');
  t.innerHTML = chat.length ? chat.map(bubble).join('') : `<div class="hero"><div class="big">${ico('chat').replace('class="i"', 'class="i" style="width:26px;height:26px"')}</div><h2>How can I help with finance today?</h2><p>Answers come only from your documents, with citations — or a clear "not found".</p><div class="sug">${SUGGEST.map(s => `<button data-s="${esc(s)}">${esc(s)}</button>`).join('')}</div></div>`;
  t.scrollTop = t.scrollHeight;
}
$('#thread').onclick = e => { const s = e.target.closest('[data-s]'); if (s) { $('#q').value = s.dataset.s; ask(); } const c = e.target.closest('[data-copy]');
  if (c) { navigator.clipboard?.writeText(c.dataset.copy).then(() => toast('Copied')); } };
$('#newChat').onclick = () => { if (cur === 'agent') { achat = []; steps('idle'); $('#alog').style.display = 'none'; agentInit(); } else { chat = []; store.set('finmind.chat', []); renderChat(); } };
$('#q').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(); } });
$('#q').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'; });
$('#send').onclick = ask;
async function ask() {
  const q = $('#q').value.trim(); if (!q || $('#send').disabled) return; $('#q').value = ''; $('#q').style.height = 'auto'; $('#send').disabled = true;
  const mode = $('#mode').value, rerank = $('#rerank').value === 'true', t0 = performance.now();
  chat.push({role:'user', text:q}); const m = {role:'bot', text:'', pending:true, cites:[], mode, rerank}; chat.push(m); renderChat();
  try {
    const r = await fetch('/api/ask', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({question:q, mode, rerank, stream:true})});
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const rd = r.body.getReader(), dec = new TextDecoder(); let buf = '';
    for (;;) { const {done, value} = await rd.read(); if (done) break; buf += dec.decode(value, {stream:true}); const parts = buf.split('\n\n'); buf = parts.pop();
      for (const p of parts) { const d = JSON.parse(p.replace(/^data: /, '')); if (d.token) m.text += d.token; if (d.done) { m.cites = d.citations; m.kind = d.kind; } }
      renderChat(); }
    m.declined = m.kind === 'declined';
  } catch (e) { m.text = 'Sorry — ' + e.message; m.declined = true; toast(e.message, true); }
  m.pending = false; m.ms = Math.round(performance.now() - t0); store.set('finmind.chat', chat.slice(-30)); renderChat(); $('#send').disabled = false; $('#q').focus();
}

// ---------- agent (Maya)
const sleep = ms => new Promise(r => setTimeout(r, ms));
const MAYA_HI = "Hi, I'm Maya from Finance Ops 👋\nI can look up overdue invoices, answer policy questions, or get payments out the door — and I'll always check with you before any money moves.\nWhat would you like to do?";
let achat = [], runId = null, runWorker = null, agentBusy = false;
const mayaAv = `<div class="av h"><svg class="face"><use href="#i-maya"/></svg></div>`;
function abubble(m, i) {
  if (m.role === 'user') return `<div class="msg user"><div class="av u">You</div><div class="bub"><div class="txt">${esc(m.text)}</div></div></div>`;
  let body = m.typing ? '<span class="typing"><span></span><span></span><span></span></span>' : `<div class="txt">${esc(m.text)}</div>`;
  if (m.approval) { const a = m.approval, p = a.proposal;
    body += `<div class="acard"><div class="ahead"><b>${p.length} payments</b><span>${money(p.reduce((s, x) => s + x.amount, 0))}</span></div>` +
      tbl(['Invoice', 'Vendor', 'Amount', 'Needs'], p.map(x => [x.id, esc(x.vendor), money(x.amount), roleBadge(x.approver_needed)]), [2]) +
      (a.state === 'open' ? `<div class="row" style="margin-top:12px"><div style="width:130px"><label class="f">Your name</label><input id="who" value="Priya"></div>
        <div style="width:170px"><label class="f">Your role</label><select id="role"><option value="clerk">Clerk</option><option value="finance_manager" selected>Finance manager</option><option value="cfo">CFO</option></select></div>
        <button class="btn pri" data-act="yes">Approve</button><button class="btn dng" data-act="no">Reject</button></div>`
        : `<div style="margin-top:10px"><span class="badge ${a.state === 'approved' ? 'b-ok' : 'b-bad'}">${a.state === 'approved' ? '✓ Approved' : '✕ Rejected'}</span></div>`) + '</div>'; }
  return `<div class="msg bot">${mayaAv}<div class="bub">${body}</div></div>`;
}
function arender() { const t = $('#athread'); t.innerHTML = achat.map(abubble).join(''); t.scrollTop = t.scrollHeight;
  $('#agentSug').style.display = achat.length > 2 ? 'none' : 'flex'; }
async function say(text, delay = 700, extra = {}) { const m = {role:'maya', typing:true}; achat.push(m); arender(); await sleep(delay);
  m.typing = false; m.text = text; Object.assign(m, extra); arender(); return m; }
function agentInit() { if (!achat.length) { achat.push({role:'maya', text:MAYA_HI}); arender(); } }
$('#agentSug').innerHTML = ['How many overdue invoices do we have?', 'Show total spend by vendor', 'Pay the overdue invoices', 'How long must financial records be retained?']
  .map(s => `<button class="btn sm" data-t="${esc(s)}">${esc(s)}</button>`).join('');
$('#agentSug').onclick = e => { const t = e.target.closest('[data-t]'); if (t) runAgent(t.dataset.t); };
$('#task').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runAgent(); } });
$('#task').addEventListener('input', e => { e.target.style.height = 'auto'; e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px'; });
$('#runBtn').onclick = () => runAgent();
$('#athread').onclick = e => { const b = e.target.closest('[data-act]'); if (b) decide(b.dataset.act === 'yes'); };
const STEPS = ['Understand', 'Look it up', 'Your approval', 'Make payments', 'Report back'];
function steps(state, worker) {  // idle | awaiting | done | rejected
  const st = state === 'idle' ? ['', '', '', '', ''] : state === 'working' ? ['done', 'active', '', '', ''] : state === 'awaiting' ? ['done', 'done', 'active', '', ''] :
    state === 'rejected' ? ['done', 'done', 'bad', 'skip', 'done'] : worker === 'payments' ? ['done', 'done', 'done', 'done', 'done'] : ['done', 'done', 'skip', 'skip', 'done'];
  const sub = ['Picks the right specialist', {rag:'Reads the policy documents', sql:'Runs a safe, read-only query', payments:'Finds overdue invoices'}[worker] || 'Documents, ledger or payments',
    'You approve or reject', 'Policy-checked and audited', 'Summarises what happened'];
  $('#steps').innerHTML = STEPS.map((n, i) => `<div class="step ${st[i]}"><div class="c">${st[i] === 'done' ? '✓' : st[i] === 'bad' ? '✕' : i + 1}</div><div><b>${n}</b><span>${sub[i]}</span></div></div>`).join('');
}
steps('idle');
async function runAgent(text) {
  const t = (text ?? $('#task').value).trim(); if (!t || agentBusy) return; agentBusy = true; $('#runBtn').disabled = true;
  $('#task').value = ''; $('#task').style.height = 'auto'; achat.push({role:'user', text:t}); arender(); steps('working');
  const think = {role:'maya', typing:true}; achat.push(think); arender();
  try {
    const [r] = await Promise.all([api('/api/agent/run', {task:t}), sleep(650)]);
    achat.splice(achat.indexOf(think), 1);
    if (r.kind === 'chat') { await say(r.answer, 300); steps('idle'); }
    else {
      runId = r.run_id; runWorker = (r.log.find(l => l.startsWith('plan')) || '').split('=')[1] || 'rag';
      $('#alog').style.display = 'block'; $('#alog').textContent = r.log.join('\n'); $('#runId').textContent = r.run_id ? 'run ' + r.run_id : '';
      await say({payments:"On it — let me pull up the overdue invoices.", sql:"Sure, let me check the ledger for you.", rag:"Good question — let me check the policy documents."}[runWorker], 500);
      if (r.status === 'awaiting_approval') {
        steps('awaiting', runWorker); const p = r.needs_approval.proposal, mm = (r.log.join(' ').match(/(\d+) of (\d+) overdue/)) || [];
        const high = p.filter(x => x.approver_needed !== 'clerk').length;
        await say(`I found ${mm[2] || p.length} overdue invoices. To start, I've lined up ${p.length} of them (${money(p.reduce((s, x) => s + x.amount, 0))} in total).` +
          (high ? ` Heads-up: ${high} of them need sign-off above clerk level.` : ' They can all be approved at clerk level.'), 900);
        await say('Shall I go ahead and pay these? I need your OK first.', 700, {approval:{proposal:p, state:'open'}});
      } else { steps('done', runWorker); await say(r.answer || "I couldn't find anything for that.", 800); }
    }
  } catch (e) { achat = achat.filter(m => !m.typing); await say("Sorry — I hit a problem: " + e.message, 300); toast(e.message, true); steps('idle'); }
  agentBusy = false; $('#runBtn').disabled = false; $('#task').focus();
}
async function decide(ok) {
  const m = achat.find(x => x.approval && x.approval.state === 'open'); if (!m || agentBusy) return;
  const who = $('#who').value || 'Someone', role = $('#role').value, roleName = {clerk:'Clerk', finance_manager:'Finance manager', cfo:'CFO'}[role];
  agentBusy = true; m.approval.state = ok ? 'approved' : 'rejected'; achat.push({role:'user', text: ok ? `Approved — ${who}, ${roleName}` : `Rejected — ${who}`}); arender();
  try { const r = await api('/api/agent/approve', {run_id:runId, approved:ok, approver:who, role});
    if (ok) { steps('done', 'payments'); const n = r.executed.length;
      await say(`Thanks ${who}, processing now…`, 600);
      await say(`All set ✅ I paid ${n} invoice${n === 1 ? '' : 's'}: ${r.executed.join(', ') || 'none'}.` +
        (r.skipped?.length ? `\n\nI couldn't pay ${r.skipped.length} (${r.skipped.join(', ')}) — as a ${roleName} you can't approve those amounts. Someone with higher authority will need to sign them off.` : '') +
        '\nEverything is recorded in the audit log.', 900);
    } else { steps('rejected', 'payments'); await say("Understood — I've cancelled this run. Nothing was paid, and I've noted the rejection in the audit log.", 700); }
    toast(ok ? 'Payments executed and audited' : 'Rejected — nothing executed');
  } catch (e) { await say("Sorry — I hit a problem: " + e.message, 300); toast(e.message, true); }
  agentBusy = false;
}

// ---------- extract
let exJson = '';
$('#exBtn').onclick = async () => { $('#exBtn').disabled = true;
  try { const j = await api('/api/extract', {text:$('#exText').value}); exJson = JSON.stringify(j, null, 2);
    $('#exOut').innerHTML = `<div class="grid g3" style="grid-template-columns:1fr 1fr 1fr;margin-bottom:12px">${Object.entries(j).map(([k, v]) => `<div class="card kpi" style="box-shadow:none"><div class="l">${esc(k)}</div><div style="font-weight:650;margin-top:4px;word-break:break-word">${esc(k === 'amount' ? money(v) : v)}</div></div>`).join('')}</div><pre class="code">${esc(exJson)}</pre>`;
  } catch (e) { toast(e.message, true); } $('#exBtn').disabled = false; };
$('#exCopy').onclick = () => exJson ? navigator.clipboard?.writeText(exJson).then(() => toast('Copied JSON')) : toast('Nothing to copy yet');

// ---------- ledger
async function loadLedger() {
  $('#kpis').innerHTML = '<div class="card kpi"><div class="sk"></div></div>'.repeat(4);
  try { const [d, od] = await Promise.all([api('/api/ledger/summary'), api('/api/ledger/overdue')]);
    const k = [['Invoices', d.invoices, 'in ledger'], ['Paid', d.paid, Math.round(100 * d.paid / d.invoices) + '% of invoices'], ['Overdue', d.overdue[0], 'pending and past due'], ['Overdue value', money(d.overdue[1]), 'needs payment']];
    $('#kpis').innerHTML = k.map(([l, v, s], i) => `<div class="card kpi"><div class="l">${l}</div><div class="v" style="${i >= 2 ? 'color:var(--warn)' : ''}">${typeof v === 'number' ? v.toLocaleString() : v}</div><div class="s">${s}</div></div>`).join('');
    const max = Math.max(...d.by_vendor.map(v => v[1]));
    $('#vendors').innerHTML = d.by_vendor.map(([v, t]) => `<div style="display:grid;grid-template-columns:130px 1fr 96px;gap:10px;align-items:center;margin:9px 0"><span>${esc(v)}</span><div style="height:9px;background:var(--code);border-radius:5px;overflow:hidden"><i style="display:block;height:100%;width:${100 * t / max}%;background:linear-gradient(90deg,var(--brand),var(--brand-2))"></i></div><b class="n" style="font-variant-numeric:tabular-nums">${money(t)}</b></div>`).join('');
    $('#pays').innerHTML = d.payments.length ? tbl(['Invoice', 'Amount', 'Approved by'], d.payments.map(p => [p[0], money(p[1]), esc(p[2])]), [1]) : '<div class="empty">No payments yet.<br>Approve some in the Payment agent tab.</div>';
    $('#overdue').innerHTML = tbl(['Invoice', 'Vendor', 'Due', 'Amount', 'Approver'], od.map(r => [r.id, esc(r.vendor), r.due, money(r.amount), roleBadge(r.approver)]), [3]);
    $('#audit').innerHTML = d.audit.length ? tbl(['Time', 'Actor', 'Action', 'Detail'], d.audit.map(a => [a[0].replace('T', ' '), esc(a[1]), `<span class="badge b-brand">${esc(a[2])}</span>`, `<span class="mono mute">${esc(a[3])}</span>`])) : '<div class="empty">No audit events yet.</div>';
  } catch (e) { toast(e.message, true); } }

// ---------- lab
const hue = i => `hsl(${(i * 53) % 360} 75% 55% / .22)`;
let tt; $('#tokIn').oninput = () => { clearTimeout(tt); tt = setTimeout(tokenize, 200); };
async function tokenize() { const t = $('#tokIn').value; if (!t) { $('#tokOut').innerHTML = ''; $('#tokStats').innerHTML = ''; return; }
  try { const j = await api('/api/tokenize', {text:t});
    $('#tokStats').innerHTML = [['Characters', j.chars], ['Tokens', j.ids.length], ['Bytes / token', (j.bytes / j.ids.length).toFixed(2)]].map(([l, v]) => `<span class="badge b-brand">${l}: <b>${v}</b></span>`).join('');
    $('#tokOut').innerHTML = j.tokens.map((x, i) => `<span class="tk" style="background:${hue(i)}" title="token id ${j.ids[i]}">${esc(x)}</span>`).join(''); } catch (e) { toast(e.message, true); } }
$('#temp').oninput = e => $('#tv').textContent = e.target.value; $('#topp').oninput = e => $('#pv').textContent = e.target.value;
$('#genBtn').onclick = async () => { $('#genBtn').disabled = true; $('#genOut').innerHTML = '<div class="sk" style="width:60%"></div>';
  try { const j = await api('/api/generate', {prompt:$('#genIn').value, temperature:+$('#temp').value, top_k:+$('#topk').value, top_p:+$('#topp').value, greedy:$('#greedy').value === 'true'});
    $('#genOut').innerHTML = `<pre class="code">${esc(j.text)}</pre><div class="meta"><span class="badge b-mute">${j.tokens_per_sec} tokens/sec</span><span class="badge b-mute">KV cache on</span></div>`;
  } catch (e) { $('#genOut').innerHTML = ''; toast(e.message, true); } $('#genBtn').disabled = false; };

// ---------- evaluation
function lossChart(h) {
  if (!h?.length) return '<div class="empty">No training history.</div>';
  const W = 520, H = 190, P = 28, xs = h.map(p => p.step), all = h.flatMap(p => [p.train, p.val]), mx = Math.max(...xs) || 1, lo = Math.min(...all), hi = Math.max(...all);
  const X = s => P + (W - 2 * P) * s / mx, Y = v => H - P - (H - 2 * P) * (v - lo) / ((hi - lo) || 1);
  const line = (k, c) => `<polyline fill="none" stroke="${c}" stroke-width="2.2" points="${h.map(p => X(p.step) + ',' + Y(p[k])).join(' ')}"/>` + h.map(p => `<circle cx="${X(p.step)}" cy="${Y(p[k])}" r="3" fill="${c}"/>`).join('');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><g stroke="var(--line)"><line x1="${P}" y1="${H - P}" x2="${W - P}" y2="${H - P}"/><line x1="${P}" y1="${P}" x2="${P}" y2="${H - P}"/></g>
    ${line('train', 'var(--brand)')}${line('val', 'var(--warn)')}<text x="${P}" y="14" font-size="11" fill="var(--mute)">loss ${hi.toFixed(1)} → ${lo.toFixed(1)}</text><text x="${W - P}" y="${H - 8}" font-size="11" fill="var(--mute)" text-anchor="end">step ${mx}</text></svg>
    <div class="meta"><span class="badge b-brand">● train</span><span class="badge b-warn">● validation</span></div>`;
}
async function loadEval() {
  try { const d = await api('/api/report'), ev = d.eval_report, m = d.metrics;
    if (ev) { const cols = [['correct', 'Correct'], ['ctx_recall', 'Ctx recall'], ['ctx_precision', 'Ctx precision'], ['mrr', 'MRR'], ['faithfulness', 'Faithful'], ['relevancy', 'Relevancy']];
      const best = Object.fromEntries(cols.map(([c]) => [c, Math.max(...Object.values(ev).map(x => x[c]))]));
      $('#evalT').innerHTML = `<table class="heat"><tr><th>Config</th>${cols.map(c => `<th style="text-align:center">${c[1]}</th>`).join('')}</tr>${Object.entries(ev).map(([k, x]) => `<tr><td><b>${esc(k)}</b></td>${cols.map(([c]) => `<td class="v ${x[c] === best[c] ? 'best' : ''}" style="background:color-mix(in srgb,var(--brand) ${Math.round(x[c] * 38)}%,transparent)">${x[c].toFixed(2)}</td>`).join('')}</tr>`).join('')}</table>`; }
    else $('#evalT').innerHTML = '<div class="empty">No evaluation yet — run <span class="mono">python -m finmind eval</span>.</div>';
    if (!m) { $('#loss').innerHTML = $('#quant').innerHTML = $('#phases').innerHTML = '<div class="empty">Run <span class="mono">python -m finmind train</span> first.</div>'; return; }
    $('#loss').innerHTML = lossChart(m.phase4_pretrain.history); const q = m.phase11_quant;
    $('#quant').innerHTML = tbl(['Precision', 'Size', 'Val loss', 'Tokens/s'], [['fp32', q.fp32], ['int8', q.int8], ['int4', q.int4]].map(([n, x]) => [`<b>${n}</b>`, x.mb + ' MB', x.val_loss, x.tokens_per_s]), [1, 2, 3]);
    const P = [['1', 'Foundations', `Classifier ${Math.round(m.phase1_classifier.val_accuracy * 100)}% accuracy vs ${Math.round(m.phase1_classifier.majority_baseline * 100)}% baseline`],
      ['2', 'Tokenization', `${m.phase2_tokenizer.bytes_per_token} bytes/token · vocab ${m.phase2_tokenizer.vocab_size}`], ['3', 'Transformer', `${m.phase3_model.params.toLocaleString()} parameters · ${m.phase3_model.n_layer} layers · ${m.phase3_model.n_head} heads`],
      ['4', 'LLM fundamentals', `Perplexity ${m.phase4_pretrain.val_perplexity} · KV cache ${m.phase4_kv_cache.with_cache_tok_s} vs ${m.phase4_kv_cache.no_cache_tok_s} tok/s`],
      ['8', 'Fine-tuning', `LoRA trains ${m.phase8_lora.trainable_pct}% of weights · held-out loss ${m.phase8_sft.heldout_loss_base} → ${m.phase8_sft.heldout_loss_tuned} · DPO ${m.phase8_dpo.preference_accuracy}`],
      ['11', 'Optimisation', `int8 is ${Math.round(100 * (1 - q.int8.mb / q.fp32.mb))}% smaller · int4 ${Math.round(100 * (1 - q.int4.mb / q.fp32.mb))}% smaller`],
      ['12', 'Research', `Speculative decoding acceptance ${m.phase12_speculative.acceptance_rate} · identical to greedy: ${m.phase12_speculative.identical_to_greedy ? 'yes' : 'no'}`]];
    $('#phases').innerHTML = P.map(([n, t, s]) => `<div class="phase"><div class="p">${n}</div><div><b>${t}</b><span>${s}</span></div></div>`).join('');
  } catch (e) { toast(e.message, true); } }

// boot
renderChat(); tokenize(); go(store.get('finmind.view', 'assistant'));
