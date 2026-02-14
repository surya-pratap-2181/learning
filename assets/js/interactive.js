// Interactive features for AI Interview Prep

document.addEventListener('DOMContentLoaded', () => {
  initReadingProgress();
  initStudyTracker();
  initKeyboardNav();
});

// --- Reading Progress Bar ---
function initReadingProgress() {
  const bar = document.createElement('div');
  bar.className = 'reading-progress';
  document.body.prepend(bar);

  window.addEventListener('scroll', () => {
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    bar.style.width = docHeight > 0 ? (window.scrollY / docHeight) * 100 + '%' : '0%';
  }, { passive: true });
}

// --- Study Progress Tracker (localStorage) ---
function initStudyTracker() {
  const pageTitle = document.querySelector('h1')?.textContent?.trim() || document.title;
  const pagePath = window.location.pathname;
  const mainContent = document.querySelector('.main-content-wrap, .main-content');
  if (!mainContent) return;

  // Only show on content pages (not home or category stubs)
  if (mainContent.textContent.split(/\s+/).length < 200) return;

  const readPages = JSON.parse(localStorage.getItem('readPages') || '{}');

  const tracker = document.createElement('div');
  tracker.className = 'progress-tracker';

  // Mark as read button
  const btn = document.createElement('button');
  const isRead = readPages[pagePath];
  btn.className = 'mark-read-btn' + (isRead ? ' is-read' : '');
  btn.textContent = isRead ? 'Studied' : 'Mark as Studied';
  btn.addEventListener('click', () => {
    if (readPages[pagePath]) {
      delete readPages[pagePath];
      btn.className = 'mark-read-btn';
      btn.textContent = 'Mark as Studied';
    } else {
      readPages[pagePath] = { title: pageTitle, date: new Date().toISOString().split('T')[0] };
      btn.className = 'mark-read-btn is-read';
      btn.textContent = 'Studied';
    }
    localStorage.setItem('readPages', JSON.stringify(readPages));
    updateProgressBadge(badge, readPages);
  });

  // Progress badge
  const badge = document.createElement('div');
  badge.className = 'progress-badge';
  updateProgressBadge(badge, readPages);
  badge.addEventListener('click', () => showProgressModal(readPages));

  tracker.appendChild(btn);
  tracker.appendChild(badge);
  document.body.appendChild(tracker);
}

function updateProgressBadge(badge, readPages) {
  const total = 60;
  const read = Object.keys(readPages).length;
  const pct = Math.round((read / total) * 100);
  const c = 2 * Math.PI * 9;
  const offset = c - (pct / 100) * c;

  badge.innerHTML = `
    <svg class="progress-ring" viewBox="0 0 24 24">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2" opacity="0.15"/>
      <circle cx="12" cy="12" r="9" fill="none" stroke="#7c3aed" stroke-width="2.5"
        stroke-dasharray="${c}" stroke-dashoffset="${offset}"
        transform="rotate(-90 12 12)" stroke-linecap="round"/>
    </svg>
    ${read}/${total}
  `;
}

function showProgressModal(readPages) {
  document.querySelector('.progress-modal-overlay')?.remove();

  const overlay = document.createElement('div');
  overlay.className = 'progress-modal-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:9999;display:flex;align-items:center;justify-content:center;';

  const read = Object.keys(readPages).length;
  const pct = Math.round((read / 60) * 100);

  const readList = Object.entries(readPages)
    .sort((a, b) => b[1].date.localeCompare(a[1].date))
    .map(([path, info]) => `<li style="margin:0.3rem 0;"><a href="${path}" style="color:#a78bfa;text-decoration:none;">${info.title}</a> <span style="opacity:0.35;font-size:0.75rem;">${info.date}</span></li>`)
    .join('');

  const modal = document.createElement('div');
  modal.style.cssText = 'background:var(--body-background-color,#1e1e2e);color:var(--body-text-color,#cdd6f4);border-radius:12px;padding:1.5rem;max-width:420px;width:90%;max-height:70vh;overflow-y:auto;border:1px solid rgba(255,255,255,0.08);';
  modal.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
      <span style="font-weight:600;">Progress</span>
      <button onclick="this.closest('.progress-modal-overlay').remove()" style="background:none;border:none;color:inherit;font-size:1.25rem;cursor:pointer;opacity:0.5;">&times;</button>
    </div>
    <div style="text-align:center;margin-bottom:1rem;">
      <div style="font-size:1.75rem;font-weight:700;color:#7c3aed;">${pct}%</div>
      <div style="font-size:0.8rem;opacity:0.5;">${read} of 60 guides</div>
      <div style="background:rgba(255,255,255,0.06);border-radius:999px;height:4px;margin-top:0.75rem;overflow:hidden;">
        <div style="background:#7c3aed;height:100%;width:${pct}%;border-radius:999px;"></div>
      </div>
    </div>
    ${read > 0 ? `<ul style="font-size:0.85rem;padding:0;margin:0;list-style:none;">${readList}</ul>` : '<p style="text-align:center;opacity:0.4;font-size:0.85rem;">No guides studied yet.</p>'}
    <div style="margin-top:1rem;text-align:center;">
      <button onclick="if(confirm('Reset all progress?')){localStorage.removeItem('readPages');location.reload();}" style="background:none;border:1px solid rgba(239,68,68,0.2);color:#ef4444;border-radius:6px;padding:0.3rem 0.75rem;cursor:pointer;font-size:0.75rem;opacity:0.6;">Reset</button>
    </div>
  `;

  overlay.appendChild(modal);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
}

// --- Keyboard Navigation ---
function initKeyboardNav() {
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key) {
      case '/':
        e.preventDefault();
        document.querySelector('.search-input')?.focus();
        break;
      case 'Escape':
        document.activeElement?.blur();
        document.querySelector('.progress-modal-overlay')?.remove();
        break;
    }
  });

  // Minimal keyboard hint
  const hint = document.createElement('div');
  hint.className = 'kbd-hint';
  hint.innerHTML = '<kbd>/</kbd> search';
  document.body.appendChild(hint);
}
