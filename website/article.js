const sectionLinks = [...document.querySelectorAll('.contents a')];
const sections = sectionLinks.map(link => document.querySelector(link.hash));
const topLink = document.querySelector('.back-to-top');
let scheduled = false;
let activeHash = null;

function updateReadingPosition() {
  scheduled = false;
  const readingLine = window.innerHeight * 0.25;
  let current = sections[0];
  for (const section of sections) {
    if (section.getBoundingClientRect().top <= readingLine) current = section;
  }
  for (const link of sectionLinks) {
    const active = current && link.hash === `#${current.id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  }
  const activeLink = sectionLinks.find(link => link.classList.contains('active'));
  if (activeLink && activeHash !== activeLink.hash) {
    activeHash = activeLink.hash;
    const nav = activeLink.parentElement;
    if (window.matchMedia('(max-width: 1050px)').matches) {
      const linkBox = activeLink.getBoundingClientRect();
      const navBox = nav.getBoundingClientRect();
      nav.scrollLeft += linkBox.left - navBox.left - (navBox.width - linkBox.width) / 2;
    }
  }
  if (topLink) topLink.hidden = window.scrollY < 600;
}

function scheduleReadingPosition() {
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(updateReadingPosition);
}

window.addEventListener('scroll', scheduleReadingPosition, { passive: true });
window.addEventListener('resize', scheduleReadingPosition);
window.addEventListener('load', scheduleReadingPosition);
updateReadingPosition();

// Native controls keep playback usable when automatic playback is disabled.
const heroVideo = document.querySelector('[data-hero-video]');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const saveData = navigator.connection?.saveData;

if (heroVideo && 'IntersectionObserver' in window) {
  let triedAutoplay = false;
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (!entry.isIntersecting) {
        heroVideo.pause();
        continue;
      }
      if (triedAutoplay || reducedMotion.matches || saveData) continue;
      triedAutoplay = true;
      heroVideo.muted = true;
      heroVideo.play().catch(() => {
        // If autoplay is blocked, the poster and play control remain usable.
      });
    }
  }, { threshold: 0.25 });
  observer.observe(heroVideo);
  reducedMotion.addEventListener('change', event => {
    if (event.matches) heroVideo.pause();
  });
}
