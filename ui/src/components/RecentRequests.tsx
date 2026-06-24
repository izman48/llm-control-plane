import type { RecentRow } from "../types";
import { GLOSSARY } from "../glossary";
import { InfoTip } from "./InfoTip";

interface Props {
  rows: RecentRow[];
}

export function RecentRequests({ rows }: Props) {
  return (
    <div className="panel">
      <h3>Recent requests</h3>
      <table className="recent-table">
        <thead>
          <tr>
            <th>req</th>
            <th>worker</th>
            <th>strategy<InfoTip text={GLOSSARY.strategy} label="What is strategy?" /></th>
            <th>TTFT<InfoTip text={GLOSSARY.ttft} label="What is TTFT?" /></th>
            <th>E2E<InfoTip text={GLOSSARY.e2e} label="What is E2E?" /></th>
            <th>tok</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.req_id}>
              <td>{r.req_id}</td>
              <td>{r.worker_id}</td>
              <td>{r.strategy}</td>
              <td>{r.ttft_s.toFixed(2)}</td>
              <td>{r.e2e_s.toFixed(2)}</td>
              <td>{r.tokens}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
