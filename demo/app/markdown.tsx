import React from "react";

/**
 * Markdown-lite: fenced code blocks, inline code, bold/italic, headings,
 * blockquotes, unordered/ordered lists, paragraphs. Deliberately tiny — no
 * dependencies, no HTML passthrough (everything is rendered as React text
 * nodes, so nothing user- or model-supplied can inject markup).
 */

let key = 0;
const k = () => `n${key++}`;

function inline(text: string): React.ReactNode[] {
  const out: React.ReactNode[] = [];
  // `code` | **bold** | *italic* / _italic_
  const re = /(`[^`\n]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|(_[^_\n]+_)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("`")) out.push(<code key={k()}>{tok.slice(1, -1)}</code>);
    else if (tok.startsWith("**")) out.push(<strong key={k()}>{tok.slice(2, -2)}</strong>);
    else out.push(<em key={k()}>{tok.slice(1, -1)}</em>);
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export function Markdown({ text, caret }: { text: string; caret?: boolean }) {
  const blocks: React.ReactNode[] = [];
  const lines = text.split("\n");
  let i = 0;

  const flushParagraph = (buf: string[]) => {
    if (buf.length === 0) return;
    blocks.push(<p key={k()}>{inline(buf.join("\n"))}</p>);
    buf.length = 0;
  };

  const para: string[] = [];

  while (i < lines.length) {
    const line = lines[i];

    // fenced code block (unterminated fences render as-is, which is what a
    // half-streamed block looks like)
    const fence = /^\s*```(\w*)\s*$/.exec(line);
    if (fence) {
      flushParagraph(para);
      const code: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) code.push(lines[i++]);
      i++; // consume closing fence if present
      blocks.push(
        <pre key={k()}>
          <code>{code.join("\n")}</code>
        </pre>,
      );
      continue;
    }

    if (line.trim() === "") {
      flushParagraph(para);
      i++;
      continue;
    }

    const heading = /^(#{1,6})\s+(.*)$/.exec(line);
    if (heading) {
      flushParagraph(para);
      const level = Math.min(heading[1].length, 3);
      const Tag = (`h${level}` as "h1" | "h2" | "h3");
      blocks.push(<Tag key={k()}>{inline(heading[2])}</Tag>);
      i++;
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flushParagraph(para);
      const quote: string[] = [];
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        quote.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      blocks.push(<blockquote key={k()}>{inline(quote.join("\n"))}</blockquote>);
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      flushParagraph(para);
      const items: string[] = [];
      while (i < lines.length && /^\s*[-*+]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*+]\s+/, ""));
        i++;
      }
      blocks.push(
        <ul key={k()}>
          {items.map((it) => (
            <li key={k()}>{inline(it)}</li>
          ))}
        </ul>,
      );
      continue;
    }

    if (/^\s*\d+[.)]\s+/.test(line)) {
      flushParagraph(para);
      const items: string[] = [];
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ""));
        i++;
      }
      blocks.push(
        <ol key={k()}>
          {items.map((it) => (
            <li key={k()}>{inline(it)}</li>
          ))}
        </ol>,
      );
      continue;
    }

    para.push(line);
    i++;
  }
  flushParagraph(para);

  if (caret) {
    const last = blocks[blocks.length - 1];
    if (React.isValidElement(last) && last.type === "p") {
      const props = last.props as { children?: React.ReactNode };
      blocks[blocks.length - 1] = (
        <p key={k()}>
          {props.children}
          <span className="caret" />
        </p>
      );
    } else {
      blocks.push(
        <p key={k()}>
          <span className="caret" />
        </p>,
      );
    }
  }

  return <>{blocks}</>;
}
