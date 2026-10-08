import { useEffect, useId, useState } from "react";
import type { RunnerAction, RunnerState } from "./RunnerPanel";

export interface SweepAxis {
  name: string;
  values: string[];
  units: string;
  numeric: boolean;
}

interface SweepNavigationProps {
  state: RunnerState;
  axes: SweepAxis[];
  onCommand: (command: RunnerAction) => void;
}

export function SweepNavigation({ state, axes, onCommand }: SweepNavigationProps) {
  const id = useId();
  const [indices, setIndices] = useState(state.selected_indices);
  const serverSelection = state.selected_indices.join(",");
  useEffect(() => setIndices(state.selected_indices), [serverSelection]);
  const selectedIndex = axes.reduce((index, axis, dimension) =>
    index * axis.values.length + (indices[dimension] ?? 0), 0);
  const pending = indices.join(",") !== serverSelection;
  const canRewind = state.can_navigate && !pending && selectedIndex <= state.completed;
  const derivedValues = Object.entries(state.selected_values)
    .filter(([name]) => !axes.some(axis => axis.name === name));

  function select(dimension: number, index: number) {
    const next = [...indices];
    next[dimension] = index;
    setIndices(next);
    onCommand({ command: "select", indices: next });
  }

  return (
    <details className="xmsr-navigation" open={state.can_navigate}>
      <summary>Browse / resume from here</summary>
      {!state.can_navigate && <p className="xmsr-muted">Pause acquisition to browse or rewind.</p>}
      {state.reopened && <p>Reopened saved run · {state.completed} committed steps</p>}
      <fieldset disabled={!state.can_navigate}>
        <legend>Parameter set</legend>
        {axes.map((axis, dimension) => (
          <div className="xmsr-coordinate" key={axis.name}>
            <label htmlFor={`${id}-${dimension}`}>{axis.name}</label>
            {axis.numeric ? (
              <input id={`${id}-${dimension}`} aria-label={`${axis.name} index`} type="range"
                min={0} max={axis.values.length - 1} step={1} value={indices[dimension] ?? 0}
                onChange={event => select(dimension, Number(event.target.value))} />
            ) : (
              <select id={`${id}-${dimension}`} aria-label={axis.name} value={indices[dimension] ?? 0}
                onChange={event => select(dimension, Number(event.target.value))}>
                {axis.values.map((value, index) => <option key={index} value={index}>{value}</option>)}
              </select>
            )}
            <output>{axis.values[indices[dimension] ?? 0]} {axis.units}</output>
            <span className="xmsr-muted">index {indices[dimension] ?? 0} / {axis.values.length - 1}</span>
          </div>
        ))}
      </fieldset>
      <p aria-live="polite">
        Selected point {selectedIndex + 1} / {state.total} · {pending ? "Loading selection…"
          : state.selection_recorded ? "Recorded data available" : "No valid recorded data"}
      </p>
      {derivedValues.length > 0 && <dl className="xmsr-derived">
        {derivedValues.map(([name, value]) => <div key={name}><dt>{name}</dt><dd>{value}</dd></div>)}
      </dl>}
      <div className="xmsr-actions">
        <button disabled={!state.can_navigate} onClick={() => onCommand("show_latest")}>Show latest</button>
        <button disabled={!canRewind} onClick={() => onCommand({ command: "rewind", indices, revision: state.revision })}>
          Resume from selected point
        </button>
      </div>
      <p className="xmsr-muted">Resuming here remeasures this point and invalidates everything after it.</p>
    </details>
  );
}
