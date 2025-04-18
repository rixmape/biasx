---
hide:
  - navigation
  - toc
---

<div class="hero-section">
  <div class="hero-content">
    <h1 class="hero-title">BiasX</h1>
    <p class="hero-subtitle">Explainable Gender Bias Analysis for<br>Face Classification</p>

    <div class="hero-buttons">
      <a href="installation/" class="md-button md-button--primary">Get Started</a>
      <a href="https://github.com/rixmape/biasx" class="md-button">View on GitHub</a>
    </div>
  </div>
</div>

<div class="terminal-demo">
  <div class="terminal-header">
    <span class="terminal-button red"></span>
    <span class="terminal-button yellow"></span>
    <span class="terminal-button green"></span>
    <span class="terminal-title">bash</span>
  </div>
  <div class="terminal-body">
    <div class="line">
      <span class="prompt">$</span>
      <span class="command">pip install biasx</span>
    </div>
    <div class="line">
      <div class="progress-bar">
        <div class="progress"></div>
      </div>
      <span class="progress-text">0%</span>
    </div>
  </div>
</div>

<div class="feature-cards">
  <div class="feature-card">
    <div class="feature-icon">
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M9.5,3A6.5,6.5 0 0,1 16,9.5C16,11.11 15.41,12.59 14.44,13.73L14.71,14H15.5L20.5,19L19,20.5L14,15.5V14.71L13.73,14.44C12.59,15.41 11.11,16 9.5,16A6.5,6.5 0 0,1 3,9.5A6.5,6.5 0 0,1 9.5,3M9.5,5C7,5 5,7 5,9.5C5,12 7,14 9.5,14C12,14 14,12 14,9.5C14,7 12,5 9.5,5Z" />
      </svg>
    </div>
    <h3>Feature-Level<br>Bias Detection</h3>
    <p>Identify exactly which facial features contribute to gender misclassifications</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M5,3H19A2,2 0 0,1 21,5V19A2,2 0 0,1 19,21H5A2,2 0 0,1 3,19V5A2,2 0 0,1 5,3M9,17V10H7V17H9M11,17V7H13V17H11M15,17V13H17V17H15Z" />
      </svg>
    </div>
    <h3>Comprehensive<br>Metrics</h3>
    <p>Quantify bias through traditional fairness metrics and feature-based analyses</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M12,9A3,3 0 0,0 9,12A3,3 0 0,0 12,15A3,3 0 0,0 15,12A3,3 0 0,0 12,9M12,17A5,5 0 0,1 7,12A5,5 0 0,1 12,7A5,5 0 0,1 17,12A5,5 0 0,1 12,17M12,4.5C7,4.5 2.73,7.61 1,12C2.73,16.39 7,19.5 12,19.5C17,19.5 21.27,16.39 23,12C21.27,7.61 17,4.5 12,4.5Z" />
      </svg>
    </div>
    <h3>Visual<br>Explanations</h3>
    <p>Generate heatmaps to reveal how facial features influence model predictions</p>
  </div>

  <div class="feature-card">
    <div class="feature-icon">
      <svg viewBox="0 0 24 24" width="24" height="24">
        <path fill="currentColor" d="M16,6L18.29,8.29L13.41,13.17L9.41,9.17L2,16.59L3.41,18L9.41,12L13.41,16L19.71,9.71L22,12V6H16Z" />
      </svg>
    </div>
    <h3>Actionable<br>Insights</h3>
    <p>Transform bias measurements into concrete model improvements</p>
  </div>
</div>

## What is BiasX?

BiasX is a comprehensive Python framework for detecting, measuring, and explaining gender bias in facial classification models. Unlike traditional fairness tools that only quantify bias through statistical metrics, BiasX reveals *why* bias occurs by connecting model decisions to specific facial features.

By combining advanced visual explanation techniques like Grad-CAM with facial landmark detection, BiasX pinpoints which facial regions (eyes, nose, lips, etc.) disproportionately influence misclassifications across gender groups.

<div class="workflow-diagram">
  <div class="workflow-step">
    <div class="step-icon">1</div>
    <div class="step-content">
      <h3>Upload Model</h3>
      <p>Connect to your trained facial classification model</p>
    </div>
  </div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">
    <div class="step-icon">2</div>
    <div class="step-content">
      <h3>Explain Predictions</h3>
      <p>Create activation maps and detect facial landmarks</p>
    </div>
  </div>
  <div class="workflow-arrow">→</div>
  <div class="workflow-step">
    <div class="step-icon">3</div>
    <div class="step-content">
      <h3>Quantify Bias</h3>
      <p>Measure feature-specific bias scores and disparities</p>
    </div>
  </div>
</div>

## Why Use BiasX?

- **Explainable Metrics:** Traditional fairness metrics tell you *if* bias exists, BiasX tells you *why* and *where* it appears.
- **Actionable Insights:** Pinpointing problematic features provides clear directions for model improvement.
- **Interpretable Results:** Visual explanations make bias findings accessible to both technical and non-technical stakeholders.
- **Research Framework:** Designed for both practical applications and academic research on algorithmic fairness.

<div class="cta-section">
  <h2>Ready to understand your model's biases?</h2>
  <div class="cta-buttons">
    <a href="installation/" class="md-button md-button--primary">Get Started</a>
    <a href="https://github.com/rixmape/biasx" class="md-button">View on GitHub</a>
  </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const progress = document.querySelector('.progress');
  const progressText = document.querySelector('.progress-text');
  let percent = 0;

  function increaseProgress() {
    if (percent < 100) {
      percent += Math.floor(Math.random() * 10) + 1;
      if (percent > 100) percent = 100;

      progress.style.width = percent + '%';
      progressText.textContent = percent + '%';

      const delay = percent < 80 ? (Math.random() * 200 + 100) : (Math.random() * 500 + 300);
      setTimeout(increaseProgress, delay);
    } else {
      progressText.textContent = 'Successfully installed biasx';
    }
  }

  setTimeout(increaseProgress, 800);
});
</script>
