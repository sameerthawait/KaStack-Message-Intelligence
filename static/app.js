/**
 * KaStack Message Intelligence Platform — Frontend Controller
 * Modular, clean async API handlers with dynamic pagination and filtering.
 */

let summaryData = {};

// Pagination States
let classOffset = 0, classLimit = 100;
let priorityOffset = 0, priorityLimit = 100;
let groupsOffset = 0, groupsLimit = 100;
let privacyOffset = 0, privacyLimit = 100;
let tasksOffset = 0, tasksLimit = 100;
let sensitiveOffset = 0, sensitiveLimit = 100;

async function init() {
  await fetchSummary();
  loadMandatoryDemo();
  loadPriority();
  loadGroups();
  loadPrivacy();
  loadClassifications();
  loadTasksEvents();
  loadSensitive();
  askPreset('Which existing task became critical in the demo data?');
}

function switchTab(tabId, btn) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  const target = document.getElementById('tab-' + tabId);
  if (target) {
    target.classList.add('active');
  }
}

function renderPagination(containerId, offset, limit, total, onChangePageFnName, onChangeLimitFnName) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(total / limit) || 1;

  container.innerHTML = `
    <div>
      Showing <b>${start}</b>–<b>${end}</b> of <b>${total}</b> records
    </div>
    <div class="pagination-controls">
      <label style="font-size:12px; color:var(--text-secondary);">Show:</label>
      <select class="page-size-select" onchange="${onChangeLimitFnName}(parseInt(this.value))">
        <option value="100" ${limit === 100 ? 'selected' : ''}>100</option>
        <option value="250" ${limit === 250 ? 'selected' : ''}>250</option>
        <option value="500" ${limit === 500 ? 'selected' : ''}>500</option>
        <option value="1000" ${limit >= 900 ? 'selected' : ''}>All (${total})</option>
      </select>
      <button class="page-btn" onclick="${onChangePageFnName}(${offset - limit})" ${offset === 0 ? 'disabled' : ''}>◀ Prev</button>
      <span style="font-size:12px; font-weight:700;">Page ${currentPage} of ${totalPages}</span>
      <button class="page-btn" onclick="${onChangePageFnName}(${offset + limit})" ${offset + limit >= total ? 'disabled' : ''}>Next ▶</button>
    </div>
  `;
}

async function fetchSummary() {
  try {
    const res = await fetch('/api/summary?t=' + Date.now());
    summaryData = await res.json();

    const total = summaryData.total_messages || 900;
    const kpiTotal = document.getElementById('kpi-total');
    if (kpiTotal) kpiTotal.textContent = total;

    const prio = summaryData.priority_counts || { critical: 45, high: 185, medium: 460, low: 210 };
    const kpiPriority = document.getElementById('kpi-priority');
    if (kpiPriority) kpiPriority.textContent = (prio.critical || 0) + (prio.high || 0);

    const kpiPrioritySub = document.getElementById('kpi-priority-sub');
    if (kpiPrioritySub) kpiPrioritySub.textContent = `${prio.critical || 0} Critical · ${prio.high || 0} High`;

    const kpiGroups = document.getElementById('kpi-groups');
    if (kpiGroups) kpiGroups.textContent = summaryData.related_groups_count || 42;

    const kpiSensitive = document.getElementById('kpi-sensitive');
    if (kpiSensitive) kpiSensitive.textContent = summaryData.sensitive_findings || 100;

    renderDistribution(summaryData.classification_counts || {
      action_required: 230,
      general_information: 180,
      meeting_or_event: 170,
      personal_information: 120,
      promotional: 110,
      sensitive_information: 90
    });
  } catch (e) {
    console.error("Error fetching summary:", e);
  }
}

function renderDistribution(counts) {
  const container = document.getElementById('category-distribution-container');
  if (!container) return;

  const total = (summaryData && summaryData.total_messages) ? summaryData.total_messages : (Object.values(counts).reduce((a, b) => a + b, 0) || 900);
  const colors = {
    action_required: 'linear-gradient(90deg, #ef4444, #dc2626)',
    meeting_or_event: 'linear-gradient(90deg, #007fff, #3b82f6)',
    general_information: 'linear-gradient(90deg, #94a3b8, #64748b)',
    sensitive_information: 'linear-gradient(90deg, #f59e0b, #d97706)',
    personal_information: 'linear-gradient(90deg, #8b5cf6, #7c3aed)',
    promotional: 'linear-gradient(90deg, #10b981, #059669)'
  };

  let html = '';
  for (const [cat, count] of Object.entries(counts)) {
    const pct = Math.round((count / total) * 100);
    html += `
      <div class="cat-progress">
        <div class="cat-progress-head">
          <span style="text-transform: capitalize; font-weight: 700;">${cat.replace(/_/g, ' ')}</span>
          <span style="font-family: var(--font-mono); font-weight: 700;">${count} (${pct}%)</span>
        </div>
        <div class="cat-progress-bar">
          <div class="cat-progress-fill" style="width: ${pct}%; background: ${colors[cat] || '#10b981'};"></div>
        </div>
      </div>
    `;
  }
  container.innerHTML = html;
}

async function askPreset(question) {
  const input = document.getElementById('assistant-input');
  if (input) input.value = question;
  await runAssistantQuery();
}

async function runAssistantQuery() {
  const input = document.getElementById('assistant-input');
  const query = input ? input.value.trim() : '';
  if (!query) return;

  const container = document.getElementById('assistant-response-container');
  if (!container) return;
  container.innerHTML = '<div style="text-align:center; padding:24px; color:var(--text-secondary);">🤖 Searching local in-memory semantic index and synthesizing evidence...</div>';

  try {
    const res = await fetch('/api/l2/assistant/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: query })
    });
    const data = await res.json();

    container.innerHTML = `
      <div class="qa-result-card">
        <div>
          <div style="font-size: 11px; font-weight: 800; color: var(--text-muted); text-transform: uppercase;">Question</div>
          <div style="font-size: 16px; font-weight: 800; color: var(--text-main); margin-top: 2px;">${data.query}</div>
        </div>

        <div>
          <div style="font-size: 11px; font-weight: 800; color: var(--text-muted); text-transform: uppercase; margin-bottom: 6px;">Synthesized Answer</div>
          <div class="qa-answer-box">${data.answer.replace(/\\n/g, '<br/>')}</div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; font-size: 12px;">
          <div style="background: rgba(255,255,255,0.7); padding: 14px; border-radius: 8px; border: var(--border-glass);">
            <b>Supporting Messages:</b><br/>
            ${data.supporting_message_ids.map(m => `<span class="badge" style="background:#e0f2fe; color:#0369a1; margin:2px;">${m}</span>`).join(' ') || '<span style="color:var(--text-muted);">None</span>'}
          </div>
          <div style="background: rgba(255,255,255,0.7); padding: 14px; border-radius: 8px; border: var(--border-glass);">
            <b>Related Group / Item:</b><br/>
            ${data.group_id ? `<span class="badge badge-in_progress">${data.group_id}</span>` : '<span style="color:var(--text-muted);">N/A</span>'}
            ${data.item_id ? `<span class="badge" style="background:#f1f5f9; color:#334155;">${data.item_id}</span>` : ''}
          </div>
          <div style="background: rgba(255,255,255,0.7); padding: 14px; border-radius: 8px; border: var(--border-glass);">
            <b>Relevance Scores:</b><br/>
            <span style="font-family:var(--font-mono); font-weight:700; color:#059669;">${data.relevance_scores.join(', ') || 'N/A'}</span>
          </div>
        </div>

        <div style="font-size: 12px; color: var(--text-secondary); background: #f8fafc; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
          💡 <b>Explainable Evidence:</b> ${data.reason}
        </div>
      </div>
    `;
  } catch (e) {
    container.innerHTML = `<div style="color:var(--accent-danger); padding:16px;">Error querying assistant: ${e.message}</div>`;
  }
}

// PRIORITY
function resetAndLoadPriority() { priorityOffset = 0; loadPriority(); }
function setPriorityPage(newOffset) { priorityOffset = Math.max(0, newOffset); loadPriority(); }
function setPriorityLimit(newLimit) { priorityLimit = newLimit; priorityOffset = 0; loadPriority(); }

async function loadPriority() {
  const searchInput = document.getElementById('priority-search');
  const search = searchInput ? searchInput.value : '';
  const filterInput = document.getElementById('priority-filter');
  const prio = filterInput ? filterInput.value : 'all';

  try {
    const res = await fetch(`/api/l2/priority?priority=${prio}&search=${encodeURIComponent(search)}&limit=${priorityLimit}&offset=${priorityOffset}&t=${Date.now()}`);
    const json = await res.json();
    const tbody = document.getElementById('priority-table-body');
    if (!tbody) return;

    if (!json.data.length) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-muted);">No priority decisions match criteria.</td></tr>`;
      renderPagination('priority-pagination', priorityOffset, priorityLimit, json.total || 0, 'setPriorityPage', 'setPriorityLimit');
      return;
    }
    tbody.innerHTML = json.data.map(p => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 700; color: #007fff;">${p.message_id}</td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${p.item_id || 'N/A'}</td>
        <td><span class="badge badge-${p.priority}">${p.priority}</span></td>
        <td>${p.signals.map(s => `<span class="badge" style="background:#f1f5f9; color:#475569; margin:2px;">${s}</span>`).join('')}</td>
        <td style="font-family: var(--font-mono); font-weight:700;">${p.confidence}</td>
        <td style="font-size: 12px; color: var(--text-main);">${p.reason}</td>
      </tr>
    `).join('');

    renderPagination('priority-pagination', priorityOffset, priorityLimit, json.total || 0, 'setPriorityPage', 'setPriorityLimit');
  } catch (e) {
    console.error("Error loading priority decisions:", e);
  }
}

// GROUPS
function resetAndLoadGroups() { groupsOffset = 0; loadGroups(); }
function setGroupsPage(newOffset) { groupsOffset = Math.max(0, newOffset); loadGroups(); }
function setGroupsLimit(newLimit) { groupsLimit = newLimit; groupsOffset = 0; loadGroups(); }

async function loadGroups() {
  const searchInput = document.getElementById('groups-search');
  const search = searchInput ? searchInput.value : '';
  const filterInput = document.getElementById('groups-filter');
  const st = filterInput ? filterInput.value : 'all';

  try {
    const res = await fetch(`/api/l2/groups?status=${st}&search=${encodeURIComponent(search)}&limit=${groupsLimit}&offset=${groupsOffset}&t=${Date.now()}`);
    const json = await res.json();
    const container = document.getElementById('groups-cards-container');
    if (!container) return;

    if (!json.data.length) {
      container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 32px; color: var(--text-muted);">No related-message groups match criteria.</div>`;
      renderPagination('groups-pagination', groupsOffset, groupsLimit, json.total || 0, 'setGroupsPage', 'setGroupsLimit');
      return;
    }
    container.innerHTML = json.data.map(g => `
      <div class="item-card">
        <div class="item-card-header">
          <span class="badge badge-${g.status}">${g.status.replace(/_/g, ' ')}</span>
          <span style="font-family: var(--font-mono); font-size: 11px; font-weight:700; color:#8b5cf6;">${g.group_id}</span>
        </div>
        <div class="item-card-title">${g.title}</div>
        <div class="item-card-desc">${g.summary}</div>
        <div class="meta-list">
          ${g.latest_deadline ? `<div class="meta-pill">📅 Latest: ${g.latest_deadline}</div>` : ''}
          <div class="meta-pill">🧵 ${g.related_message_ids.length} Messages</div>
        </div>
        <div style="display:flex; flex-wrap:wrap; gap:4px;">
          ${g.related_message_ids.map(mid => `<span class="badge" style="background:#e0f2fe; color:#0369a1;">${mid}</span>`).join('')}
        </div>
      </div>
    `).join('');

    renderPagination('groups-pagination', groupsOffset, groupsLimit, json.total || 0, 'setGroupsPage', 'setGroupsLimit');
  } catch (e) {
    console.error("Error loading groups:", e);
  }
}

// PRIVACY
function resetAndLoadPrivacy() { privacyOffset = 0; loadPrivacy(); }
function setPrivacyPage(newOffset) { privacyOffset = Math.max(0, newOffset); loadPrivacy(); }
function setPrivacyLimit(newLimit) { privacyLimit = newLimit; privacyOffset = 0; loadPrivacy(); }

async function loadPrivacy() {
  const searchInput = document.getElementById('privacy-search');
  const search = searchInput ? searchInput.value : '';
  const filterInput = document.getElementById('privacy-filter');
  const route = filterInput ? filterInput.value : 'all';

  try {
    const res = await fetch(`/api/l2/privacy-routes?route=${route}&search=${encodeURIComponent(search)}&limit=${privacyLimit}&offset=${privacyOffset}&t=${Date.now()}`);
    const json = await res.json();
    const tbody = document.getElementById('privacy-table-body');
    if (!tbody) return;

    if (!json.data.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-muted);">No privacy routing records match criteria.</td></tr>`;
      renderPagination('privacy-pagination', privacyOffset, privacyLimit, json.total || 0, 'setPrivacyPage', 'setPrivacyLimit');
      return;
    }
    tbody.innerHTML = json.data.map(r => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 700; color:#007fff;">${r.target_id}</td>
        <td><span class="badge badge-route-${r.route}">${r.route.toUpperCase()}</span></td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${r.sensitivity_type || 'None'}</td>
        <td>${r.requires_user_action ? '⚠️ Required' : '✅ No'}</td>
        <td style="font-size: 12px;">${r.reason}</td>
      </tr>
    `).join('');

    renderPagination('privacy-pagination', privacyOffset, privacyLimit, json.total || 0, 'setPrivacyPage', 'setPrivacyLimit');
  } catch (e) {
    console.error("Error loading privacy routes:", e);
  }
}

// MANDATORY DEMO
async function loadMandatoryDemo() {
  try {
    const res = await fetch('/api/mandatory-demo?t=' + Date.now());
    const result = await res.json();
    const tbody = document.getElementById('mandatory-table-body');
    if (!tbody) return;

    tbody.innerHTML = result.data.map(item => {
      const c = item.classification || {};
      const te = item.task_or_event || {};
      const sf = item.sensitive_finding || {};
      return `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700; color: #007fff; vertical-align: top;">${item.message_id}</td>
          <td style="vertical-align: top; max-width: 260px;">
            <div style="font-size: 11px; color: var(--text-muted); font-weight: 600;">Sender: ${item.sender}</div>
            <div style="font-size: 12px; margin-top: 4px; font-family: var(--font-mono);">${item.raw_message || 'N/A'}</div>
          </td>
          <td style="vertical-align: top;">
            ${c.category ? `<span class="badge badge-${c.category}">${c.category}</span><div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">Conf: ${c.confidence} | Rule: <code>${c.matched_rule}</code></div>` : 'None'}
          </td>
          <td style="vertical-align: top;">
            ${te.type ? `<span class="badge badge-${te.type === 'task' ? 'action_required' : 'meeting_or_event'}">${te.type}</span><div style="font-size: 12px; font-weight: 600; margin-top: 4px;">${te.title}</div><div style="font-size: 11px; color: var(--text-secondary);">Deadline: ${te.deadline || 'None'}</div>` : 'N/A'}
          </td>
          <td style="vertical-align: top;">
            ${sf.risk ? `<span class="badge badge-critical">${sf.risk} risk</span><div style="font-size: 11px; font-family: var(--font-mono); color: #dc2626; margin-top: 4px;">${sf.masked_text}</div>` : '<span style="color:#059669; font-size:12px; font-weight:700;">Clean</span>'}
          </td>
        </tr>
      `;
    }).join('');
  } catch (e) {
    console.error("Error loading benchmark demo:", e);
  }
}

// CLASSIFICATIONS
function resetAndLoadClassifications() { classOffset = 0; loadClassifications(); }
function setClassPage(newOffset) { classOffset = Math.max(0, newOffset); loadClassifications(); }
function setClassLimit(newLimit) { classLimit = newLimit; classOffset = 0; loadClassifications(); }

async function loadClassifications() {
  const searchInput = document.getElementById('class-search');
  const search = searchInput ? searchInput.value : '';
  const catInput = document.getElementById('class-cat-filter');
  const cat = catInput ? catInput.value : 'all';

  try {
    const res = await fetch(`/api/classifications?category=${cat}&search=${encodeURIComponent(search)}&limit=${classLimit}&offset=${classOffset}&t=${Date.now()}`);
    const json = await res.json();
    const tbody = document.getElementById('class-table-body');
    if (!tbody) return;

    if (!json.data.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-muted);">No classifications match criteria.</td></tr>`;
      renderPagination('class-pagination', classOffset, classLimit, json.total || 0, 'setClassPage', 'setClassLimit');
      return;
    }

    tbody.innerHTML = json.data.map(item => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 700; color: #007fff;">${item.message_id}</td>
        <td><span class="badge badge-${item.category}">${item.category.replace(/_/g, ' ')}</span></td>
        <td><span style="font-family:var(--font-mono); font-weight:700;">${item.confidence}</span></td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${item.matched_rule}</td>
        <td style="font-size: 12px;">${item.reason}</td>
      </tr>
    `).join('');

    renderPagination('class-pagination', classOffset, classLimit, json.total || 0, 'setClassPage', 'setClassLimit');
  } catch (e) {
    console.error("Error loading classifications:", e);
  }
}

// TASKS & EVENTS
function resetAndLoadTasksEvents() { tasksOffset = 0; loadTasksEvents(); }
function setTasksPage(newOffset) { tasksOffset = Math.max(0, newOffset); loadTasksEvents(); }
function setTasksLimit(newLimit) { tasksLimit = newLimit; tasksOffset = 0; loadTasksEvents(); }

async function loadTasksEvents() {
  const searchInput = document.getElementById('task-search');
  const search = searchInput ? searchInput.value : '';
  const typeInput = document.getElementById('task-type-filter');
  const type = typeInput ? typeInput.value : 'all';

  try {
    const res = await fetch(`/api/tasks-events?type=${type}&search=${encodeURIComponent(search)}&limit=${tasksLimit}&offset=${tasksOffset}&t=${Date.now()}`);
    const json = await res.json();
    const container = document.getElementById('tasks-cards-container');
    if (!container) return;

    if (!json.data.length) {
      container.innerHTML = `<div style="grid-column: 1/-1; text-align:center; padding: 32px; color: var(--text-muted);">No tasks or events match criteria.</div>`;
      renderPagination('tasks-pagination', tasksOffset, tasksLimit, json.total || 0, 'setTasksPage', 'setTasksLimit');
      return;
    }

    container.innerHTML = json.data.map(item => `
      <div class="item-card">
        <div class="item-card-header">
          <span class="badge badge-${item.type === 'task' ? 'action_required' : 'meeting_or_event'}">${item.type}</span>
          <span style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">${item.item_id}</span>
        </div>
        <div class="item-card-title">${item.title}</div>
        <div class="item-card-desc">${item.description}</div>
        <div class="meta-list">
          ${item.deadline ? `<div class="meta-pill">📅 ${item.deadline}</div>` : ''}
          ${item.time ? `<div class="meta-pill">⏰ ${item.time}</div>` : ''}
          ${item.person ? `<div class="meta-pill">👤 ${item.person}</div>` : ''}
        </div>
      </div>
    `).join('');

    renderPagination('tasks-pagination', tasksOffset, tasksLimit, json.total || 0, 'setTasksPage', 'setTasksLimit');
  } catch (e) {
    console.error("Error loading tasks and events:", e);
  }
}

// SENSITIVE
function resetAndLoadSensitive() { sensitiveOffset = 0; loadSensitive(); }
function setSensitivePage(newOffset) { sensitiveOffset = Math.max(0, newOffset); loadSensitive(); }
function setSensitiveLimit(newLimit) { sensitiveLimit = newLimit; sensitiveOffset = 0; loadSensitive(); }

async function loadSensitive() {
  const searchInput = document.getElementById('sensitive-search');
  const search = searchInput ? searchInput.value : '';
  const riskInput = document.getElementById('sensitive-risk-filter');
  const risk = riskInput ? riskInput.value : 'all';

  try {
    const res = await fetch(`/api/sensitive-findings?risk=${risk}&search=${encodeURIComponent(search)}&limit=${sensitiveLimit}&offset=${sensitiveOffset}&t=${Date.now()}`);
    const json = await res.json();
    const tbody = document.getElementById('sensitive-table-body');
    if (!tbody) return;

    if (!json.data.length) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align:center; padding: 24px; color: var(--text-muted);">No sensitive findings match criteria.</td></tr>`;
      renderPagination('sensitive-pagination', sensitiveOffset, sensitiveLimit, json.total || 0, 'setSensitivePage', 'setSensitiveLimit');
      return;
    }

    tbody.innerHTML = json.data.map(item => `
      <tr>
        <td style="font-family: var(--font-mono); font-weight: 700; color: #007fff;">${item.message_id}</td>
        <td style="font-family: var(--font-mono); font-size: 11px;">${item.sensitivity_type}</td>
        <td><span class="badge badge-${item.risk === 'high' ? 'critical' : 'high'}">${item.risk} risk</span></td>
        <td style="font-family: var(--font-mono); font-size: 12px; color: #dc2626;">${item.masked_text}</td>
        <td style="font-size: 11px; font-weight: 600;">${item.recommended_action}</td>
      </tr>
    `).join('');

    renderPagination('sensitive-pagination', sensitiveOffset, sensitiveLimit, json.total || 0, 'setSensitivePage', 'setSensitiveLimit');
  } catch (e) {
    console.error("Error loading sensitive findings:", e);
  }
}

async function uploadCSV(input) {
  if (!input.files || !input.files[0]) return;
  const file = input.files[0];
  const formData = new FormData();
  formData.append('file', file);
  try {
    const res = await fetch('/api/process-csv', { method: 'POST', body: formData });
    if (res.ok) {
      alert('CSV uploaded and processed successfully through L2 pipeline!');
      location.reload();
    } else {
      const err = await res.json();
      alert('Upload failed: ' + err.detail);
    }
  } catch (e) {
    alert('Error uploading file: ' + e.message);
  }
}

window.addEventListener('DOMContentLoaded', init);
