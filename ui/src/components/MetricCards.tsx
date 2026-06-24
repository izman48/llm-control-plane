import type { MetricsSnapshot } from "../types";
import { GLOSSARY } from "../glossary";
import { InfoTip } from "./InfoTip";

interface Props {
  metrics: MetricsSnapshot;
}

interface Card {
  label: string;
  value: string;
  tip: string;
}

export function MetricCards({ metrics }: Props) {
  const cards: Card[] = [
    { label: "Throughput", value: `${metrics.throughput_tok_s.toFixed(0)} tok/s`, tip: GLOSSARY.throughput },
    { label: "Offered load", value: `${metrics.offered_load_req_s.toFixed(0)} req/s`, tip: GLOSSARY.offeredLoad },
    { label: "In-flight", value: `${metrics.in_flight}`, tip: GLOSSARY.inFlight },
    { label: "Completed", value: `${metrics.completed_total}`, tip: GLOSSARY.completed },
    { label: "TTFT p50", value: `${metrics.ttft_p50_s.toFixed(2)} s`, tip: GLOSSARY.ttftP50 },
    { label: "TTFT p99", value: `${metrics.ttft_p99_s.toFixed(2)} s`, tip: GLOSSARY.ttftP99 },
    { label: "E2E p99", value: `${metrics.e2e_p99_s.toFixed(2)} s`, tip: GLOSSARY.e2eP99 },
  ];
  return (
    <div className="cards">
      {cards.map((c) => (
        <div className="card" key={c.label}>
          <div className="card-label">
            {c.label}
            <InfoTip text={c.tip} label={`What is ${c.label}?`} />
          </div>
          <div className="card-value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}
