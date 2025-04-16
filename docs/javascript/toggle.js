// theme-toggle.js
document.addEventListener('DOMContentLoaded', function() {
    // Create the theme toggle button
    const themeToggle = document.createElement('button');
    themeToggle.className = 'theme-toggle';
    themeToggle.setAttribute('aria-label', 'Toggle light/dark theme');
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
    
    // Function to toggle between light and dark mode
    function toggleTheme() {
      const currentScheme = document.body.getAttribute('data-md-color-scheme') || 'default';
      const newScheme = currentScheme === 'default' ? 'slate' : 'default';
      
      // Update the color scheme
      document.body.setAttribute('data-md-color-scheme', newScheme);
      
      // Save the preference in local storage
      localStorage.setItem('theme', newScheme);
      
      // Update the toggle button state
      updateToggleButtonState(newScheme);
    }
    
    // Function to update the toggle button state based on current theme
    function updateToggleButtonState(theme) {
      if (theme === 'slate') {
        themeToggle.classList.add('dark-mode');
      } else {
        themeToggle.classList.remove('dark-mode');
      }
    }
    
    // Check for saved theme preference or respect prefers-color-scheme
    function setInitialTheme() {
      const savedTheme = localStorage.getItem('theme');
      
      if (savedTheme) {
        // Apply saved theme
        document.body.setAttribute('data-md-color-scheme', savedTheme);
        updateToggleButtonState(savedTheme);
      } else {
        // Respect OS preference if no saved theme
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const initialTheme = prefersDark ? 'slate' : 'default';
        document.body.setAttribute('data-md-color-scheme', initialTheme);
        updateToggleButtonState(initialTheme);
      }
    }
    
    // Add click event listener to toggle theme
    themeToggle.addEventListener('click', toggleTheme);
    
    // Set initial theme
    setInitialTheme();
    
    // Listen for OS theme changes if no saved preference
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
      if (!localStorage.getItem('theme')) {
        const newTheme = e.matches ? 'slate' : 'default';
        document.body.setAttribute('data-md-color-scheme', newTheme);
        updateToggleButtonState(newTheme);
      }
    });
  });

  // Check for saved theme preference or default to dark mode
  function setInitialTheme() {
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme) {
      // Apply saved theme
      document.body.setAttribute('data-md-color-scheme', savedTheme);
      updateToggleButtonState(savedTheme);
    } else {
      // Default to dark mode (slate)
      const initialTheme = 'slate';
      document.body.setAttribute('data-md-color-scheme', initialTheme);
      updateToggleButtonState(initialTheme);
      // Save this preference
      localStorage.setItem('theme', initialTheme);
    }
  }