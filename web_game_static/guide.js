const progressBar = document.querySelector('#reading-progress-bar');
const backToTop = document.querySelector('#back-to-top');
const chapterLinks = [...document.querySelectorAll('.lesson-sidebar nav a[href^="#"]')];
const chapters = chapterLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);

let framePending = false;

function updateGuideNavigation() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const progress = scrollable > 0 ? Math.min(1, Math.max(0, window.scrollY / scrollable)) : 0;
  progressBar.style.transform = `scaleX(${progress})`;
  backToTop.classList.toggle('is-visible', window.scrollY > 560);

  const readingLine = window.scrollY + Math.min(window.innerHeight * 0.3, 240);
  let activeChapter = chapters[0];
  chapters.forEach((chapter) => {
    if (chapter.offsetTop <= readingLine) activeChapter = chapter;
  });
  chapterLinks.forEach((link) => {
    const isActive = link.getAttribute('href') === `#${activeChapter?.id}`;
    if (isActive) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
}

function requestNavigationUpdate() {
  if (framePending) return;
  framePending = true;
  window.requestAnimationFrame(() => {
    updateGuideNavigation();
    framePending = false;
  });
}

window.addEventListener('scroll', requestNavigationUpdate, { passive: true });
window.addEventListener('resize', requestNavigationUpdate);
backToTop.addEventListener('click', () => {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  window.scrollTo({ top: 0, behavior: reducedMotion ? 'auto' : 'smooth' });
});

updateGuideNavigation();
