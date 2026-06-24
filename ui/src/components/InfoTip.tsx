import { useState } from "react";

interface Props {
  /** The explanation to show. */
  text: string;
  /** Accessible name for the button; defaults to a generic label. */
  label?: string;
}

// A small "(i)" affordance that reveals a plain-English explanation on hover,
// keyboard focus, or click (tap). Dependency-free; the bubble is positioned with
// CSS relative to the inline wrapper. role="tooltip" ties it to assistive tech.
export function InfoTip({ text, label = "More info" }: Props) {
  const [open, setOpen] = useState(false);
  return (
    <span className="infotip">
      <button
        type="button"
        className="infotip-btn"
        aria-label={label}
        aria-expanded={open}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
      >
        i
      </button>
      {open && (
        <span role="tooltip" className="infotip-bubble">
          {text}
        </span>
      )}
    </span>
  );
}
