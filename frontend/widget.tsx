import { createRender, useModelState, useModel } from "@anywidget/react";
import { RunnerPanel, type RunnerState, type LogRecord } from "./RunnerPanel";
import type { SweepAxis } from "./SweepNavigation";
import "./runner.css";

function RunnerWidget() {
  const [state] = useModelState<RunnerState>("state");
  const [logs] = useModelState<LogRecord[]>("logs");
  const [axes] = useModelState<SweepAxis[]>("axes");
  const model = useModel();
  return <RunnerPanel state={state} logs={logs} axes={axes}
    onCommand={action => model.send(typeof action === "string" ? { command: action } : action)} />;
}

export default { render: createRender(RunnerWidget) };
