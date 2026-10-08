import { useState } from "react";

export type RunnerStatus = "initialized" | "running" | "paused" | "finished" | "cancelled" | "failed";
export type RunnerCommand = "start" | "pause" | "resume" | "cancel";

export interface RunnerState {
  name: string;
  status: RunnerStatus;
  completed: number;
  total: number;
  elapsed: number;
  remaining: number | null;
  rate: number;
  filename: string;
  path: string;
  error: string;
}

export interface LogRecord {
  id: number;
  level: string;
  time: string;
  message: string;
}

interface RunnerPanelProps {
  state: RunnerState;
  logs: LogRecord[];
  onCommand: (command: RunnerCommand) => void;
}

const actions: Partial<Record<RunnerStatus, { label: string; command: RunnerCommand }>> = {
  initialized: { label: "Start", command: "start" },
  running: { label: "Pause", command: "pause" },
  paused: { label: "Resume", command: "resume" },
};

function duration(seconds: number): string {
  const value = Math.max(0, Math.floor(seconds));
  return `${Math.floor(value / 60)}m ${value % 60}s`;
}

export function RunnerPanel({ state, logs, onCommand }: RunnerPanelProps) {
  const [clipboardStatus, setClipboardStatus] = useState("");
  const action = actions[state.status];
  const progress = state.total > 0 ? Math.min(100, Math.max(0, state.completed / state.total * 100)) : 0;

  async function copy(value: string) {
    try {
      await navigator.clipboard.writeText(value);
      setClipboardStatus("Copied");
    } catch {
      setClipboardStatus("Could not copy to clipboard");
    }
  }

  return (
    <section className="xmsr-runner" aria-label={`${state.name} runner`}>
      <header>
        <h3>{state.name}</h3>
        <div className="xmsr-actions">
          {action && <button onClick={() => onCommand(action.command)}>{action.label}</button>}
          {action && <button onClick={() => onCommand("cancel")}>Cancel</button>}
        </div>
      </header>
      <div className="xmsr-summary">
        <span className={`xmsr-status xmsr-status-${state.status}`} role="status">{state.status}</span>
        <span>{state.completed} / {state.total} steps · {progress.toFixed(1)}%</span>
      </div>
      <progress aria-label="Measurement progress" value={state.completed} max={Math.max(1, state.total)} />
      <div className="xmsr-timing">
        <span>Elapsed: {duration(state.elapsed)}</span>
        <span>Remaining: {state.remaining === null ? "—" : duration(state.remaining)}</span>
        <span>{state.rate.toFixed(2)} steps/s</span>
      </div>
      {state.filename && (
        <div className="xmsr-file">
          <code title={state.path}>{state.filename}</code>
          <button onClick={() => copy(state.filename)}>Copy filename</button>
          <button onClick={() => copy(state.path)}>Copy path</button>
          <span aria-live="polite">{clipboardStatus}</span>
        </div>
      )}
      {state.error && <p role="alert" className="xmsr-error">{state.error}</p>}
      <div className="xmsr-logs" role="log" aria-label="Measurement logs" aria-live="off">
        {logs.length === 0 && <p className="xmsr-empty">No logs yet.</p>}
        {[...logs].reverse().map(record => (
          <div className={`xmsr-log xmsr-log-${record.level}`} key={record.id}>
            <span>{record.time} [{record.level.toUpperCase()}]</span> {record.message}
          </div>
        ))}
      </div>
    </section>
  );
}
