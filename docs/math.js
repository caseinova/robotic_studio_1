function renderProjectMath(root = document) {
  if (!window.katex) return;
  root.querySelectorAll('.equation, .math-inline').forEach(element => {
    if (element.dataset.mathRendered) return;
    const expression = element.textContent.trim();
    const displayMode = element.classList.contains('equation');
    try {
      katex.render(expression, element, {
        displayMode,
        output: 'htmlAndMathml',
        throwOnError: true,
        strict: 'error',
        trust: false
      });
      element.dataset.mathRendered = 'true';
      if (displayMode) {
        element.tabIndex = 0;
        element.setAttribute('role', 'region');
        element.setAttribute('aria-label', 'Equation');
      }
    } catch (error) {
      element.classList.add('math-error');
      console.error('Unable to render equation:', expression, error);
    }
  });
}

renderProjectMath();
