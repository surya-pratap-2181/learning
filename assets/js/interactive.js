// Interactive features for AI Interview Prep

document.addEventListener('DOMContentLoaded', function () {
  initReadingProgress();
  initStudyTracker();
  initKeyboardNav();
});

// --- Reading Progress Bar ---
function initReadingProgress() {
  const bar = document.createElement('div');
  bar.className = 'reading-progress';
  document.body.prepend(bar);

  window.addEventListener('scroll', function () {
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

  // Mark as read button
  const btn = document.createElement('button');
  const isRead = readPages[pagePath];
  btn.className = 'mark-read-btn' + (isRead ? ' is-read' : '');
  btn.textContent = isRead ? 'Studied' : 'Mark as Studied';
  btn.addEventListener('click', function () {
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
  badge.addEventListener('click', function () { showProgressModal(readPages); });

  const tracker = document.createElement('div');
  tracker.className = 'progress-tracker';
  tracker.appendChild(btn);
  tracker.appendChild(badge);
  document.body.appendChild(tracker);
}

function updateProgressBadge(badge, readPages) {
  const total = 60;
  const read = Object.keys(readPages).length;
  const pct = Math.round((read / total) * 100);
  const circumference = 2 * Math.PI * 9;
  const offset = circumference - (pct / 100) * circumference;

  badge.innerHTML =
    '<svg class="progress-ring" viewBox="0 0 24 24">' +
      '<circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2" opacity="0.15"/>' +
      '<circle cx="12" cy="12" r="9" fill="none" stroke="#7c3aed" stroke-width="2.5"' +
        ' stroke-dasharray="' + circumference + '" stroke-dashoffset="' + offset + '"' +
        ' transform="rotate(-90 12 12)" stroke-linecap="round"/>' +
    '</svg>' +
    read + '/' + total;
}

function showProgressModal(readPages) {
  document.querySelector('.progress-modal-overlay')?.remove();

  const read = Object.keys(readPages).length;
  const pct = Math.round((read / 60) * 100);

  const readList = Object.entries(readPages)
    .sort(function (a, b) { return b[1].date.localeCompare(a[1].date); })
    .map(function (entry) {
      const path = entry[0];
      const info = entry[1];
      return '<li class="modal-list-item">' +
        '<a href="' + path + '" class="modal-link">' + info.title + '</a> ' +
        '<span class="modal-date">' + info.date + '</span></li>';
    })
    .join('');

  const listContent = read > 0
    ? '<ul class="modal-list">' + readList + '</ul>'
    : '<p class="modal-empty">No guides studied yet.</p>';

  const overlay = document.createElement('div');
  overlay.className = 'progress-modal-overlay';
  overlay.innerHTML =
    '<div class="progress-modal">' +
      '<div class="modal-header">' +
        '<span class="modal-title">Progress</span>' +
        '<button class="modal-close">&times;</button>' +
      '</div>' +
      '<div class="modal-stats">' +
        '<div class="modal-pct">' + pct + '%</div>' +
        '<div class="modal-subtitle">' + read + ' of 60 guides</div>' +
        '<div class="modal-bar"><div class="modal-bar-fill" style="width:' + pct + '%"></div></div>' +
      '</div>' +
      listContent +
      '<div class="modal-footer">' +
        '<button class="modal-reset">Reset</button>' +
      '</div>' +
    '</div>';

  overlay.querySelector('.modal-close').addEventListener('click', function () { overlay.remove(); });
  overlay.querySelector('.modal-reset').addEventListener('click', function () {
    if (confirm('Reset all progress?')) {
      localStorage.removeItem('readPages');
      location.reload();
    }
  });
  overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.remove(); });
  document.body.appendChild(overlay);
}

// --- Keyboard Navigation ---
function initKeyboardNav() {
  document.addEventListener('keydown', function (e) {
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

  const hint = document.createElement('div');
  hint.className = 'kbd-hint';
  hint.innerHTML = '<kbd>/</kbd> search';
  document.body.appendChild(hint);
}
