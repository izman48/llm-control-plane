import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { InfoTip } from "./InfoTip";

interface Props {
  title: string;
  history: number[];
  color: string;
  tip: string;
}

// A small live line chart over a rolling history buffer. Generic so the same
// component renders both the throughput and offered-load series.
export function TimeSeriesChart({ title, history, color, tip }: Props) {
  const data = history.map((v, i) => ({ t: i, v }));
  return (
    <div className="panel chart-panel">
      <h3>
        {title}
        <InfoTip text={tip} label={`About the ${title} chart`} />
      </h3>
      <div style={{ width: "100%", height: 160 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
            <XAxis dataKey="t" hide />
            <YAxis width={44} tick={{ fontSize: 11 }} />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="v"
              stroke={color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
