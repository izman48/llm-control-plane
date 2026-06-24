import { GLOSSARY } from "../glossary";
import { InfoTip } from "./InfoTip";

// Backend selector. The hosted demo runs the Sim backend; the real-model and
// OpenAI-compatible endpoint backends are self-hosted only (an endpoint can't be
// taken server-side on a public box — SSRF — and a hosted box can't reach a
// reviewer's localhost model anyway).
const BACKENDS = [
  { id: "sim", label: "Sim", enabled: true, tip: GLOSSARY.backendSim },
  { id: "endpoint", label: "Endpoint (self-hosted)", enabled: false, tip: GLOSSARY.backendEndpoint },
  { id: "realmodel", label: "Real model (self-hosted)", enabled: false, tip: GLOSSARY.backendReal },
];

export function BackendSelector() {
  return (
    <div className="panel">
      <h3>Backend</h3>
      {BACKENDS.map((b) => (
        <label key={b.id} className="row">
          <input type="radio" name="backend" defaultChecked={b.id === "sim"} disabled={!b.enabled} />
          {b.label}
          <InfoTip text={b.tip} label={`What is the ${b.label} backend?`} />
        </label>
      ))}
    </div>
  );
}
