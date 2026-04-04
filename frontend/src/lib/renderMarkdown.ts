import { marked } from 'marked';
import katex from 'katex';

// Configure marked for safe output
marked.setOptions({
  breaks: true,
  gfm: true,
});

function renderKatex(tex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(tex.trim(), { displayMode, throwOnError: false });
  } catch {
    return `<code>${tex}</code>`;
  }
}

/**
 * Render markdown + LaTeX to HTML.
 * Handles display math: $$...$$ and \[...\]
 * Handles inline math: $...$ and \(...\)
 * Then runs marked on the rest.
 */
export function renderMarkdown(text: string): string {
  // Display math: $$...$$ and \[...\]
  let processed = text.replace(/\$\$([\s\S]*?)\$\$/g, (_m, tex) => renderKatex(tex, true));
  processed = processed.replace(/\\\[([\s\S]*?)\\\]/g, (_m, tex) => renderKatex(tex, true));

  // Inline math: $...$ (not $$) and \(...\)
  processed = processed.replace(/(?<!\$)\$(?!\$)(.*?)\$/g, (_m, tex) => renderKatex(tex, false));
  processed = processed.replace(/\\\((.*?)\\\)/g, (_m, tex) => renderKatex(tex, false));

  // Run marked on the result
  return marked.parse(processed, { async: false }) as string;
}
