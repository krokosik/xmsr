import logging
from abc import ABC, abstractmethod
import xarray as xr
from PyQt5.QtCore import (
    pyqtSignal,
    QObject,
    Qt,
    QState,
    QStateMachine,
    QSignalTransition,
)
from PyQt5.QtWidgets import (
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QLabel,
    QLineEdit,
    QCheckBox,
    QFileDialog,
    QTableView,
    QSplitter,
    QSizePolicy,
)
from .measurement import _Measurement
from .metadata_model import _MetadataModel
import os
from tqdm import tqdm
from pyqtgraph.widgets.MatplotlibWidget import MatplotlibWidget
from typing import Optional


class QtLoggingHelper(ABC):
    @abstractmethod
    def transform(self, msg: str):
        raise NotImplementedError()


class QtLoggingBasic(QtLoggingHelper):
    def transform(self, msg: str):
        return msg


class QtLoggingColoredLogs(QtLoggingHelper):
    def __init__(self):
        # offensive programming: crash if necessary if import is not present
        pass

    def transform(self, msg: str):
        try:
            import coloredlogs.converter

            msg_html = coloredlogs.converter.convert(msg)
            return msg_html
        except ImportError:
            return msg


class QTextEditLogger(logging.Handler, QObject):
    appendText = pyqtSignal(str)

    def __init__(
        self,
        logger_: logging.Logger,
        formatter: logging.Formatter,
        text_widget: QPlainTextEdit,
        parent: QWidget,
    ):
        super(QTextEditLogger, self).__init__()
        super(QObject, self).__init__(parent=parent)  # type: ignore
        self.text_widget = text_widget
        self.text_widget.setReadOnly(True)
        try:
            self.helper = QtLoggingColoredLogs()
            self.appendText.connect(self.text_widget.appendHtml)
            logger_.info("Using QtLoggingColoredLogs")
        except ImportError:
            self.helper = QtLoggingBasic()
            self.appendText.connect(self.text_widget.appendPlainText)
            logger_.warning("Using QtLoggingBasic")
        self.setFormatter(formatter)
        logger_.addHandler(self)

    def emit(self, record: logging.LogRecord):
        msg = self.format(record)
        display_msg = self.helper.transform(msg=msg)
        self.appendText.emit(display_msg)


class _MeasurementGUI(QWidget):
    _param_boxes: list[QComboBox] = []
    _current_index: int = 0
    _max_indices: list[int] = []
    _togglable_widgets: list[QWidget] = []
    _with_preview: bool = True

    def __init__(self, measurement: type[_Measurement]):
        super().__init__()
        self._measurement = measurement
        self.setWindowTitle(self._measurement.__name__)

        self.setMinimumWidth(1500)
        self.setMinimumHeight(800)

        controller_splitter = QSplitter(Qt.Orientation.Vertical)

        self._setup_preview()
        self._setup_measurement_kwargs_layout()
        self._setup_params_layout()
        self._setup_progress_bar()
        self._setup_logger()

        self._setup_state_machine()

        container = QWidget()
        container.setLayout(self._measurement_kwargs_layout)
        controller_splitter.addWidget(container)
        container = QWidget()
        measurement_layout = QVBoxLayout()
        measurement_layout.addLayout(self._params_layout)
        measurement_layout.addWidget(self.plain_text_edit_logger, 1)
        measurement_layout.addWidget(self.pbar)
        container.setLayout(measurement_layout)
        controller_splitter.addWidget(container)
        controller_splitter.setStretchFactor(1, 1)

        if self._with_preview:
            splitter = QSplitter(Qt.Orientation.Horizontal)
            splitter.addWidget(controller_splitter)
            splitter.addWidget(self.mw)
        else:
            splitter = controller_splitter

        self.setStyleSheet("QSplitterHandle { background: #ccc }")

        layout = QHBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)

        self.machine.start()

    def _setup_preview(self):
        try:
            self._measurement.plot_preview(None, None, None, None)  # type: ignore
        except Exception as e:
            if isinstance(e, NotImplementedError):
                self._with_preview = False
                return

        self.mw = MatplotlibWidget(figsize=(4, 3))

    def _get_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory", self._directory_widget.text()
        )
        if directory:
            self._directory_widget.setText(directory)

    def _setup_measurement_kwargs_layout(self):
        self._measurement_kwargs_layout = QHBoxLayout()

        left_layout = QVBoxLayout()

        # Directory
        directory_label = QLabel("&Directory")
        left_layout.addWidget(directory_label)
        self._togglable_widgets.append(directory_label)

        directory_layout = QHBoxLayout()
        self._directory_widget = QLineEdit()
        directory_label.setBuddy(self._directory_widget)
        self._directory_widget.setText(
            self._measurement.target_directory or os.getcwd()
        )
        self._togglable_widgets.append(self._directory_widget)
        directory_layout.addWidget(self._directory_widget)

        self._directory_button = QPushButton("Pick")
        self._directory_button.clicked.connect(self._get_dir)
        self._togglable_widgets.append(self._directory_button)
        directory_layout.addWidget(self._directory_button)

        left_layout.addLayout(directory_layout)

        # Filename
        filename_label = QLabel("&Filename")
        self.filename_widget = QLineEdit()
        self.filename_widget.setText(self._measurement.__name__)
        filename_label.setBuddy(self.filename_widget)
        left_layout.addWidget(filename_label)
        left_layout.addWidget(self.filename_widget)
        self._togglable_widgets.append(filename_label)
        self._togglable_widgets.append(self.filename_widget)

        # Checkboxes
        self.timestamp_checkbox = QCheckBox("With &Timestamp")
        self.timestamp_checkbox.setChecked(
            getattr(self._measurement, "timestamp", True)
        )
        left_layout.addWidget(self.timestamp_checkbox)
        self._togglable_widgets.append(self.timestamp_checkbox)

        self.coords_checkbox = QCheckBox("With &Coordinates")
        self.coords_checkbox.setChecked(getattr(self._measurement, "with_coords", True))
        left_layout.addWidget(self.coords_checkbox)
        self._togglable_widgets.append(self.coords_checkbox)

        self.overwrite_checkbox = QCheckBox("&Overwrite")
        self.overwrite_checkbox.setChecked(
            getattr(self._measurement, "overwrite", False)
        )
        left_layout.addWidget(self.overwrite_checkbox)
        self._togglable_widgets.append(self.overwrite_checkbox)

        # Metadata label
        metadata_label = QLabel("Metadata")
        metadata_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(metadata_label)
        self._togglable_widgets.append(metadata_label)

        # Metadata buttons
        button_layout = QHBoxLayout()

        add_metadata_button = QPushButton("Add")
        add_metadata_button.clicked.connect(self._insert_metadata_row)
        button_layout.addWidget(add_metadata_button)
        self._togglable_widgets.append(add_metadata_button)

        remove_metadata_button = QPushButton("Del")
        remove_metadata_button.clicked.connect(self._remove_metadata_row)
        button_layout.addWidget(remove_metadata_button)
        self._togglable_widgets.append(remove_metadata_button)

        left_layout.addLayout(button_layout)

        # OK button
        self._setup_button = QPushButton(self)
        self._setup_button.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self._setup_button.setMinimumHeight(50)
        self._setup_button.setText("OK")
        left_layout.addWidget(self._setup_button)

        # Metadata table
        self.metadata_model = _MetadataModel(getattr(self._measurement, "metadata", {}))
        self.metadata_table = QTableView()
        self.metadata_table.setModel(self.metadata_model)
        self.metadata_table.horizontalHeader().setStretchLastSection(True)
        self._togglable_widgets.append(self.metadata_table)

        self._measurement_kwargs_layout.addLayout(left_layout)
        self._measurement_kwargs_layout.addWidget(self.metadata_table, 1)

    def _insert_metadata_row(self):
        self.metadata_model.insertEmptyRow()

    def _remove_metadata_row(self):
        indices = [i for i in self.metadata_table.selectedIndexes() if not i.column()]

        if len(indices):
            self.metadata_model.removeRows(indices[0].row(), len(indices))

    def _setup_params_layout(self):
        self._params_layout = QHBoxLayout()

        logging_layout = QVBoxLayout()
        self.logging_selection = QComboBox()
        self.logging_selection.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.logging_selection.setCurrentIndex(
            self.logging_selection.findText(
                logging.getLevelName(self.LOG.getEffectiveLevel())
            )
        )

        logging_label = QLabel("&Logging")
        logging_label.setBuddy(self.logging_selection)
        logging_layout.addWidget(logging_label)
        logging_layout.addWidget(self.logging_selection)
        self._params_layout.addLayout(logging_layout)
        self.logging_selection.currentTextChanged.connect(self._change_logging_level)

        for name, values in self._measurement._param_coords.items():
            label = QLabel(name)
            combo = QComboBox()
            combo.addItems([str(v) for v in values])
            combo.setEditable(True)

            self._param_boxes.append(combo)

            widget = QVBoxLayout()
            widget.addWidget(label)
            widget.addWidget(combo)
            self._params_layout.addLayout(widget)

        self.go_pause_button = QPushButton(self)
        self.go_pause_button.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )

        self._params_layout.addWidget(self.go_pause_button)

    def _setup_progress_bar(self):
        self.pbar = QProgressBar()
        self.pbar.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _setup_logger(self):
        self.plain_text_edit_logger = QPlainTextEdit(self)
        LOG_FMT = "{asctime} | {levelname:10s} | {message}"
        try:
            import coloredlogs

            FORMATTER = coloredlogs.ColoredFormatter(fmt=LOG_FMT, style="{")
        except ImportError:
            FORMATTER = logging.Formatter(fmt=LOG_FMT, style="{")

        self.logging_ = QTextEditLogger(
            logger_=self.LOG,
            formatter=FORMATTER,
            text_widget=self.plain_text_edit_logger,
            parent=self,
        )

    @property
    def LOG(self) -> logging.Logger:
        return logging.getLogger(self._measurement.__name__)

    def _change_logging_level(self, name: str):
        logging_level = logging._nameToLevel[name]
        self.LOG.setLevel(level=logging_level)
        logging.basicConfig(level=logging_level)

    def _setup_state_machine(self):
        self.machine = QStateMachine(self)
        self.setup = QState()
        self.measure = QState()
        self.go = QState(self.measure)
        self.pause = QState(self.measure)
        self.measure.setInitialState(self.pause)

        self.setup.addTransition(self._setup_button.clicked, self.measure)
        self.setup.assignProperty(self._setup_button, "text", "OK")
        self.setup.assignProperty(self.go_pause_button, "enabled", False)
        self.setup.assignProperty(self.pbar, "visible", False)

        self.measure.entered.connect(self._setup_measurement)
        self.measure.entered.connect(lambda: self.measurement_thread.start())

        self.measure.assignProperty(self._setup_button, "text", "Reset")
        self.measure.assignProperty(self.go_pause_button, "enabled", True)
        self.measure.assignProperty(self.pbar, "visible", True)
        for widget in self._togglable_widgets:
            self.measure.assignProperty(widget, "enabled", False)
            self.setup.assignProperty(widget, "enabled", True)

        for combo in self._param_boxes:
            self.go.assignProperty(combo, "enabled", False)
            self.setup.assignProperty(combo, "enabled", False)
            self.pause.assignProperty(combo, "enabled", True)
            self.setup.assignProperty(combo, "currentIndex", 0)
            trans = QSignalTransition(combo.currentIndexChanged)
            trans.triggered.connect(self._refresh_param_options)
            trans.triggered.connect(self._redraw_preview)
            self.pause.addTransition(trans)

        self.go.assignProperty(self.go_pause_button, "text", "Pause")
        self.go.addTransition(self.go_pause_button.clicked, self.pause)
        self.go.entered.connect(lambda: self.measurement_thread.toggle_pause.emit())

        self.pause.assignProperty(self.go_pause_button, "text", "Run")
        self.pause.addTransition(self.go_pause_button.clicked, self.go)
        self.pause.entered.connect(lambda: self.measurement_thread.toggle_pause.emit())
        self.pause.entered.connect(self._refresh_param_options)
        self.pause.exited.connect(self._revert_progress)

        self.measure.addTransition(self._setup_button.clicked, self.setup)
        self.measure.exited.connect(self._destroy_measurement)

        self.machine.addState(self.setup)
        self.machine.addState(self.measure)
        self.machine.setInitialState(self.setup)

    def _setup_measurement(self):
        self.measurement_thread = self._measurement(
            target_directory=self._directory_widget.text(),
            filename=self.filename_widget.text(),
            timestamp=self.timestamp_checkbox.isChecked(),
            overwrite=self.overwrite_checkbox.isChecked(),
            with_coords=self.coords_checkbox.isChecked(),
            metadata=self.metadata_model.items,
        )
        self.measurement_thread._is_single_run = False

        self.measurement_thread.start_signal.connect(self._prepare_progress_bar)
        self.measurement_thread.step_signal.connect(self._update_progress_bar)
        self.measurement_thread.new_result_signal.connect(self._redraw_preview)
        self.measurement_thread.finished_signal.connect(self._finish_progress_bar)
        self.go.addTransition(self.measurement_thread.finished_signal, self.pause)

        self._max_indices = list(
            range(len(self.measurement_thread._param_coords.keys()))
        )

        self._change_logging_level(self.logging_selection.currentText())

    def _destroy_measurement(self):
        self.measurement_thread.stop.emit()
        self.measurement_thread.wait()
        del self.measurement_thread

    def _update_progress(self, data: dict):
        self._current_index = data["n"]
        indices = data["indices"]

        data["ncols"] = 0
        self.pbar.setFormat(tqdm.format_meter(**data))

        for i in range(len(self._max_indices)):
            self._max_indices[i] = indices[i]
            if self._max_indices[i] != self._param_boxes[i].currentIndex():
                self._param_boxes[i].setCurrentIndex(self._max_indices[i])

    def _prepare_progress_bar(self, data: dict):
        self.pbar.setRange(0, data["total"])
        self.pbar.setValue(data["n"])

        self._update_progress(data)
        self._refresh_param_options()

    def _update_progress_bar(self, data: dict):
        self.pbar.setValue(data["n"])
        self._update_progress(data)

    def _finish_progress_bar(self):
        self.LOG.info(
            f"Measurement finished! Open with\n\nxr.open_{'dataarray' if len(self.measurement_thread.variables) == 1 else 'dataset'}('{self.measurement_thread._path}', engine='zarr')\n"
        )

    def _refresh_param_options(self, _index: int = 0):
        previous_index_full = False

        for param_box, max_index in zip(self._param_boxes, self._max_indices):
            if param_box.currentIndex() > max_index and not previous_index_full:
                param_box.setCurrentIndex(max_index)

            for i in range(param_box.count()):
                param_box.model().item(i).setEnabled(  # type: ignore
                    previous_index_full or i <= max_index
                )

            if param_box.currentIndex() < max_index:
                previous_index_full = True

    @property
    def _current_indices(self) -> tuple[int, ...]:
        return tuple((param_box.currentIndex() for param_box in self._param_boxes))

    def _redraw_preview(self, data: Optional[xr.DataArray | xr.Dataset] = None):
        self.LOG.debug(f"Redrawing preview for measurement no. {self._current_indices}")
        if not self._with_preview:
            return

        if data is None:
            self.LOG.debug(f"Getting data for measurement no. {self._current_indices}")
            data = self.measurement_thread.get_measurement_by_indices(
                self._current_indices
            )

        try:
            self.LOG.debug(
                f"Plotting preview for measurement no. {self._current_indices}"
            )

            self.mw.getFigure().clear()
            self.measurement_thread.plot_preview(
                data,
                self.measurement_thread.result,
                self.mw.getFigure().subplots(),
            )
            self.mw.draw()
        except Exception as e:
            self.LOG.warning(f"Failed to plot preview: {e}")

    def _revert_progress(self):
        self.LOG.debug(f"Reverting progress to measurement no. {self._current_indices}")
        self.measurement_thread.revert_progress.emit(
            tuple((param_box.currentIndex() for param_box in self._param_boxes))
        )
