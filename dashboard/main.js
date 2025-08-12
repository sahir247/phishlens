const API_BASE = "http://127.0.0.1:8000";

function fmtTs(ts){
  const d = new Date(ts*1000);
  return d.toLocaleString();
}

async function loadEvents(limit){
  const res = await fetch(`${API_BASE}/events?limit=${limit||100}`);
  if(!res.ok) throw new Error('Failed to fetch events');
  return res.json();
}

function renderRows(rows, filter){
  const tbody = document.getElementById('rows');
  tbody.innerHTML = '';
  const f = (filter||'').toLowerCase();
  rows.filter(r => {
    if(!f) return true;
    const txt = (r.url + ' ' + (r.reasons||[]).join(' ')).toLowerCase();
    return txt.includes(f);
  }).forEach(r => {
    const tr = document.createElement('tr');
    const riskPct = Math.round((r.risk_score||0)*100);
    tr.innerHTML = `
      <td>${fmtTs(r.ts)}</td>
      <td><a href="${r.url}" target="_blank">${r.url}</a></td>
      <td><span class="risk ${riskPct>=80?'red':riskPct>=50?'amber':'green'}">${riskPct}%</span></td>
      <td>${(r.reasons||[]).join('; ')}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function refresh(){
  const limit = parseInt(document.getElementById('limit').value||'100',10);
  const filter = document.getElementById('filter').value||'';
  const rows = await loadEvents(limit);
  renderRows(rows, filter);
}

document.getElementById('refresh').onclick = refresh;
window.addEventListener('load', refresh);
