// Set the saved theme before the stylesheet loads. First visits use the light reading theme.
(() => {
  const storageKey = 'trackmaniarl-article-theme';
  let theme = 'light';
  try {
    const saved = localStorage.getItem(storageKey);
    if (saved === 'light' || saved === 'dark') theme = saved;
  } catch {
    // Reading remains available when browser storage is disabled.
  }
  document.documentElement.dataset.theme = theme;

  function renderTheme() {
    document.documentElement.dataset.theme = theme;
    document.querySelectorAll('[data-theme-toggle]').forEach(button => {
      const next = theme === 'dark' ? 'light' : 'dark';
      button.querySelector('[data-theme-label]').textContent = `${next === 'light' ? 'Light' : 'Dark'} mode`;
      button.setAttribute('aria-label', `Switch to ${next} mode`);
      button.hidden = false;
    });
    document.querySelectorAll('[data-light-href]').forEach(link => {
      link.href = theme === 'dark' ? link.dataset.darkHref : link.dataset.lightHref;
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-theme-toggle]').forEach(button => {
      button.addEventListener('click', () => {
        theme = theme === 'dark' ? 'light' : 'dark';
        renderTheme();
        try {
          localStorage.setItem(storageKey, theme);
        } catch {
          // The switch still works for this visit without persistent storage.
        }
      });
    });
    renderTheme();
  });
})();
