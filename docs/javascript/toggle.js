document.addEventListener("DOMContentLoaded", function () {
  const themeToggle = document.createElement("button");
  themeToggle.className = "theme-toggle";
  themeToggle.setAttribute("aria-label", "Toggle light/dark theme");
  themeToggle.innerHTML = `
      <svg class="light-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      </svg>
      <svg class="dark-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      </svg>
    `;

  document.body.appendChild(themeToggle);

  function toggleTheme() {
    const currentScheme = document.body.getAttribute("data-md-color-scheme") || "default";
    const newScheme = currentScheme === "default" ? "slate" : "default";

    document.body.setAttribute("data-md-color-scheme", newScheme);
    localStorage.setItem("theme", newScheme);

    updateToggleButtonState(newScheme);
  }

  function updateToggleButtonState(theme) {
    if (theme === "slate") {
      themeToggle.classList.add("dark-mode");
    } else {
      themeToggle.classList.remove("dark-mode");
    }
  }

  themeToggle.addEventListener("click", toggleTheme);

  setInitialTheme();

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
    if (!localStorage.getItem("theme")) {
      const newTheme = e.matches ? "slate" : "default";
      document.body.setAttribute("data-md-color-scheme", newTheme);
      updateToggleButtonState(newTheme);
    }
  });
});

function setInitialTheme() {
  const savedTheme = localStorage.getItem("theme");

  if (savedTheme) {
    document.body.setAttribute("data-md-color-scheme", savedTheme);
    updateToggleButtonState(savedTheme);
  } else {
    const initialTheme = "slate";
    document.body.setAttribute("data-md-color-scheme", initialTheme);
    updateToggleButtonState(initialTheme);

    localStorage.setItem("theme", initialTheme);
  }
}
