/* deepcrew-ai docs — main.js */

// ─── Scroll-triggered section reveal ────────────────────────
const observer = new IntersectionObserver(
  (entries) => entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); }),
  { threshold: 0.08, rootMargin: '0px 0px -40px 0px' }
);
document.querySelectorAll('.section').forEach(s => observer.observe(s));

// ─── Active sidebar link on scroll ──────────────────────────
const sections = [...document.querySelectorAll('[data-section]')];
const sidebarLinks = [...document.querySelectorAll('.sidebar-link[data-target]')];

const activateLink = (id) => {
  sidebarLinks.forEach(l => {
    l.classList.toggle('active', l.dataset.target === id);
  });
};

const scrollSpy = new IntersectionObserver(
  (entries) => {
    entries.forEach(e => { if (e.isIntersecting) activateLink(e.target.dataset.section); });
  },
  { threshold: 0, rootMargin: '-20% 0px -70% 0px' }
);
sections.forEach(s => scrollSpy.observe(s));

// ─── Sidebar link click smooth scroll ───────────────────────
sidebarLinks.forEach(link => {
  link.addEventListener('click', () => {
    const target = document.getElementById(link.dataset.target);
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    sidebarLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  });
});

// ─── Clipboard helper ────────────────────────────────────────
// navigator.clipboard.writeText() requires the document to still be focused
// and requires "recent" user activation. Both can be lost across an awaited
// fetch() (slow network, devtools open, focus stolen by an extension, an
// embedding iframe without a clipboard-write permission), which throws
// NotAllowedError even though the user did click the button. Fall back to
// the legacy execCommand('copy') textarea trick, and if even that is denied
// (blocked entirely in some sandboxed/embedded contexts), fall back one more
// time to a native window.prompt() with the text pre-filled and selected —
// that's a plain browser dialog, not the Clipboard API, so no permission or
// focus requirement can block it; the user copies it out with Ctrl/Cmd+C
// themselves. Returns 'clipboard' | 'exec' | 'prompt' | 'cancelled'.
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return 'clipboard';
  } catch (err) {
    console.warn('[deepcrew] clipboard.writeText failed, trying execCommand:', err);
  }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    if (ok) return 'exec';
    console.warn('[deepcrew] execCommand("copy") returned false, falling back to prompt()');
  } catch (err) {
    console.warn('[deepcrew] execCommand("copy") failed, falling back to prompt()', err);
  }
  // Last resort: a native dialog the user copies from manually. Always available.
  const result = window.prompt('Copy this text (Ctrl/Cmd+C, then Enter or Esc):', text);
  return result === null ? 'cancelled' : 'prompt';
}

function copyButtonLabel(outcome, successLabel) {
  if (outcome === 'clipboard' || outcome === 'exec') return successLabel;
  if (outcome === 'prompt') return 'Copied via dialog';
  return 'Copy failed';
}

// ─── Copy buttons ────────────────────────────────────────────
document.querySelectorAll('.code-copy, .copy-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const target = btn.dataset.copy
      ? document.getElementById(btn.dataset.copy)
      : btn.closest('.code-block')?.querySelector('code');
    const text = target?.textContent?.trim() || btn.dataset.text || '';
    const original = btn.innerHTML;
    const outcome = await copyText(text);
    if (outcome === 'cancelled') return;
    btn.classList.add('copied');
    btn.innerHTML = copyButtonLabel(outcome, 'Copied');
    setTimeout(() => { btn.innerHTML = original; btn.classList.remove('copied'); }, 2000);
  });
});

// ─── Copy page as Markdown (for pasting into an LLM) ──────────
// The markdown is embedded inline in every page as <script type="text/plain"
// class="page-md">. Reading it from the DOM works everywhere — including a page
// opened directly from disk via file:// (where fetch() of a local .md is blocked
// by the browser and was the cause of the recurring "Copy failed"). fetch(data-md)
// is kept only as a fallback for any page that lacks the inline block.
async function getPageMarkdown(btn) {
  const inline = document.querySelector('script.page-md');
  if (inline && inline.textContent.trim()) return inline.textContent.trim();
  const src = btn.dataset.md;
  if (!src) return '';
  try {
    const res = await fetch(src);
    if (!res.ok) throw new Error('fetch failed: ' + res.status);
    return (await res.text()).trim();
  } catch (err) {
    console.warn('[deepcrew] copy-page-btn fetch fallback failed:', err);
    return '';
  }
}

document.querySelectorAll('.copy-page-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const original = btn.innerHTML;
    const text = await getPageMarkdown(btn);
    if (!text) {
      btn.innerHTML = 'Copy failed';
      setTimeout(() => { btn.innerHTML = original; }, 2000);
      return;
    }
    const outcome = await copyText(text);
    if (outcome === 'cancelled') return;
    btn.classList.add('copied');
    btn.innerHTML = copyButtonLabel(outcome, 'Copied for LLM');
    setTimeout(() => { btn.innerHTML = original; btn.classList.remove('copied'); }, 2000);
  });
});

// ─── Copy an AI-ready implementation prompt for this feature ──
document.querySelectorAll('.copy-prompt-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const id = btn.dataset.prompt;
    const el = id ? document.getElementById(id) : null;
    const text = el?.textContent?.trim() || '';
    if (!text) return;
    const original = btn.innerHTML;
    const outcome = await copyText(text);
    if (outcome === 'cancelled') return;
    btn.classList.add('copied');
    btn.innerHTML = copyButtonLabel(outcome, 'Prompt copied');
    setTimeout(() => { btn.innerHTML = original; btn.classList.remove('copied'); }, 2000);
  });
});

// ─── Nav link active state on scroll ────────────────────────
const navLinks = document.querySelectorAll('.nav-links a[data-section]');
navLinks.forEach(link => {
  link.addEventListener('click', () => {
    navLinks.forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  });
});

// ─── Animated number counters ───────────────────────────────
const counterObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el = entry.target;
      const target = parseInt(el.dataset.count, 10);
      const duration = 1200;
      const start = performance.now();
      const tick = (now) => {
        const progress = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(ease * target).toLocaleString();
        if (progress < 1) requestAnimationFrame(tick);
        else el.textContent = el.dataset.suffix
          ? target.toLocaleString() + el.dataset.suffix
          : target.toLocaleString();
      };
      requestAnimationFrame(tick);
      counterObserver.unobserve(el);
    });
  },
  { threshold: 0.5 }
);
document.querySelectorAll('[data-count]').forEach(el => counterObserver.observe(el));

// ─── Mobile sidebar toggle ───────────────────────────────────
const toggle = document.querySelector('.menu-toggle');
const sidebar = document.querySelector('aside');

// On mobile the primary top-nav links (Docs / Features / Examples / API) are hidden
// (.nav-links { display:none }), which left Features and Examples unreachable from the
// drawer. Clone them into a section at the top of the drawer — this reuses each page's
// own relative hrefs automatically. The section is display:none on desktop (where the
// top nav is already visible) and revealed only inside the drawer at <=768px.
if (sidebar) {
  const navLinks = document.querySelector('.nav-links');
  if (navLinks && !sidebar.querySelector('.sidebar-navlinks')) {
    const section = document.createElement('div');
    section.className = 'sidebar-section sidebar-navlinks';
    const label = document.createElement('span');
    label.className = 'sidebar-label';
    label.textContent = 'Navigate';
    section.appendChild(label);
    navLinks.querySelectorAll('a').forEach(a => {
      const link = document.createElement('a');
      link.className = 'sidebar-link';
      link.setAttribute('href', a.getAttribute('href'));
      link.textContent = a.textContent.trim();
      section.appendChild(link);
    });
    sidebar.insertBefore(section, sidebar.firstChild);
  }
}

// Inject backdrop element.
// IMPORTANT: it must live INSIDE .page-layout, not on <body>. .page-layout is a
// stacking context (position:relative; z-index:1), so the <aside> drawer's z-index
// is only meaningful within that context. A body-level backdrop would paint on top
// of the entire page-layout subtree — including the drawer — dimming it. Placing the
// backdrop as a sibling of the drawer inside page-layout lets it dim <main> while the
// drawer (higher z-index within the same context) stays above it.
const backdrop = document.createElement('div');
backdrop.className = 'sidebar-backdrop';
(document.querySelector('.page-layout') || document.body).appendChild(backdrop);

const ICON_MENU = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M1 4h14M1 8h14M1 12h14"/></svg>';
const ICON_CLOSE = '<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><path d="M2 2l12 12M14 2L2 14"/></svg>';

function closeSidebar() {
  sidebar.style.display = '';
  backdrop.classList.remove('open');
  if (toggle) toggle.innerHTML = ICON_MENU;
}
function openSidebar() {
  sidebar.style.display = 'block';
  backdrop.classList.add('open');
  if (toggle) toggle.innerHTML = ICON_CLOSE;
}

if (toggle && sidebar) {
  toggle.addEventListener('click', () => {
    sidebar.style.display === 'block' ? closeSidebar() : openSidebar();
  });
  backdrop.addEventListener('click', closeSidebar);
  // Close sidebar when a link is tapped on mobile
  sidebar.querySelectorAll('.sidebar-link').forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 768) closeSidebar();
    });
  });
}

// ─── Code block line highlighting on hover ─────────────────
document.querySelectorAll('pre code').forEach(code => {
  const lines = code.innerHTML.split('\n');
  if (lines.length < 3) return;
  code.innerHTML = lines
    .map((line, i) => `<span class="code-line" data-line="${i+1}">${line}</span>`)
    .join('\n');
  code.querySelectorAll('.code-line').forEach(line => {
    line.addEventListener('mouseenter', () => line.style.background = 'rgba(255,255,255,0.04)');
    line.addEventListener('mouseleave', () => line.style.background = '');
  });
});

// ─── Prism.js re-highlight after init ────────────────────────
if (window.Prism) Prism.highlightAll();

// ─── Agent Diagram State Machine (JS inline-style, immune to document.hidden) ──
(function() {
  const d = document.getElementById('agent-diagram');
  if (!d) return;

  // Refs
  const orch      = d.querySelector('.diag-card--orch');
  const pulse     = d.querySelector('.diag-pulse-ring');
  const stRouting = d.querySelector('.diag-st--routing');
  const stRouted  = d.querySelector('.diag-st--routed');
  const wiresD    = [...d.querySelectorAll('.diag-svg--down .diag-wire')];
  const agents    = [...d.querySelectorAll('.diag-card--agent')];
  const dots      = [...d.querySelectorAll('.diag-agent-dot')];
  const bars      = [...d.querySelectorAll('.diag-gen-bar')];
  const done      = [...d.querySelectorAll('.diag-done-tag')];
  const wiresU    = [...d.querySelectorAll('.diag-svg--up .diag-wire')];
  const synth     = d.querySelector('.diag-card--synth');
  const glowRing  = d.querySelector('.diag-glow-ring');
  const check     = d.querySelector('.diag-synth-check');

  function s(el, props) { if (el) Object.assign(el.style, props); }
  function sa(arr, props) { arr.forEach(el => s(el, props)); }
  function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

  function reset() {
    s(orch,  { opacity: '0', transform: 'translateY(-8px)' });
    s(pulse, { opacity: '0' });
    s(stRouting, { opacity: '0' });
    s(stRouted,  { opacity: '0' });
    sa(wiresD, { strokeDashoffset: '130', opacity: '0' });
    sa(agents, { opacity: '0', transform: 'translateY(6px)' });
    sa(dots,   { background: 'rgba(255,255,255,0.2)', boxShadow: 'none' });
    sa(bars,   { width: '0' });
    sa(done,   { opacity: '0' });
    sa(wiresU, { strokeDashoffset: '130', opacity: '0' });
    s(synth,     { opacity: '0', transform: 'translateY(8px)' });
    s(glowRing,  { opacity: '0', boxShadow: 'none' });
    s(check,     { opacity: '0', transform: 'scale(0.7)' });
  }

  async function cycle() {
    reset();
    await wait(350);

    // 1. Orchestrator
    s(orch, { opacity: '1', transform: 'none' });
    await wait(850);

    // 2. Pulse ring flash
    s(pulse, { opacity: '0.5' });
    await wait(220);
    s(pulse, { opacity: '0' });

    // 3. Routing status
    s(stRouting, { opacity: '1' });
    await wait(700);

    // 4. Down wires draw (staggered via setTimeout inside async)
    wiresD.forEach((w, i) => setTimeout(() => s(w, { strokeDashoffset: '0', opacity: '1' }), i * 90));
    await wait(750);

    // 5. Agent cards appear (staggered)
    agents.forEach((a, i) => setTimeout(() => s(a, { opacity: '1', transform: 'none' }), i * 140));
    await wait(600);

    // 6. Switch routing → dispatched
    s(stRouting, { opacity: '0' });
    s(stRouted,  { opacity: '1' });
    await wait(350);

    // 7. Generation starts — dots pulse, bars grow staggered per --td
    sa(dots, { background: '#fff', boxShadow: '0 0 5px rgba(255,255,255,0.6)' });
    bars.forEach(b => {
      const tw = getComputedStyle(b).getPropertyValue('--tw').trim() || '70%';
      const td = parseFloat(getComputedStyle(b).getPropertyValue('--td')) * 1000 || 0;
      setTimeout(() => s(b, { width: tw }), td);
    });
    await wait(2600);

    // 8. Generation done — bars retract, badges appear, dots dim
    sa(bars, { width: '0' });
    sa(dots, { background: 'rgba(255,255,255,0.2)', boxShadow: 'none' });
    sa(done, { opacity: '1' });
    await wait(650);

    // 9. Up wires draw
    wiresU.forEach((w, i) => setTimeout(() => s(w, { strokeDashoffset: '0', opacity: '1' }), i * 90));
    await wait(750);

    // 10. Synthesizer appears + glows
    s(synth, { opacity: '1', transform: 'none' });
    await wait(400);
    s(glowRing, { opacity: '1', boxShadow: '0 0 0 1px rgba(255,255,255,0.28), 0 0 22px rgba(255,255,255,0.1)' });
    await wait(500);
    s(check, { opacity: '1', transform: 'scale(1)' });

    // 11. Hold → restart
    await wait(1400);
    setTimeout(cycle, 100);
  }

  cycle();
})();
