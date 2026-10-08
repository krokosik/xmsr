import { createRender, useModelState, useModel } from "@anywidget/react";
import { RunnerPanel, type RunnerState, type LogRecord } from "./RunnerPanel";
import "./runner.css";

function RunnerWidget() {
  const [state] = useModelState<RunnerState>("state");
  const [logs] = useModelState<LogRecord[]>("logs");
  const model = useModel();
  return <RunnerPanel state={state} logs={logs} onCommand={command => model.send({ command })} />;
}

export default { render: createRender(RunnerWidget) };
