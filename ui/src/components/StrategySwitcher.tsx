import { GLOSSARY } from "../glossary";
import { InfoTip } from "./InfoTip";

interface Props {
  strategies: string[];
  current: string;
  onChange: (name: string) => void;
}

export function StrategySwitcher({ strategies, current, onChange }: Props) {
  return (
    <div className="panel">
      <h3>
        Routing strategy
        <InfoTip text={GLOSSARY.strategy} label="What is a routing strategy?" />
      </h3>
      <select
        aria-label="routing strategy"
        value={current}
        onChange={(e) => onChange(e.target.value)}
      >
        {strategies.map((s) => (
          <option key={s} value={s}>
            {s}
          </option>
        ))}
      </select>
    </div>
  );
}
