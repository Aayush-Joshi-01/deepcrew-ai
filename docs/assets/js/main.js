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

// ─── Copy buttons ────────────────────────────────────────────
document.querySelectorAll('.code-copy, .copy-btn').forEach(btn => {
  btn.addEventListener('click', async () => {
    const target = btn.dataset.copy
      ? document.getElementById(btn.dataset.copy)
      : btn.closest('.code-block')?.querySelector('code');
    const text = target?.textContent?.trim() || btn.dataset.text || '';
    try {
      await navigator.clipboard.writeText(text);
      const original = btn.innerHTML;
      btn.classList.add('copied');
      btn.innerHTML = btn.classList.contains('copy-btn')
        ? '✓ Copied' : '✓&nbsp;Copied';
      setTimeout(() => { btn.innerHTML = original; btn.classList.remove('copied'); }, 2000);
    } catch {}
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
if (toggle && sidebar) {
  toggle.addEventListener('click', () => {
    const open = sidebar.style.display === 'block';
    sidebar.style.display = open ? '' : 'block';
    sidebar.style.zIndex = '200';
  });
  document.addEventListener('click', (e) => {
    if (!sidebar.contains(e.target) && e.target !== toggle) {
      sidebar.style.display = '';
    }
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
