import numpy as np
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex


class _MetadataModel(QAbstractTableModel):
    _items: list[list]

    def __init__(self, data: dict, helper_globals: dict = {}):
        super().__init__()
        self.items = data
        self.globals = helper_globals.update({"np": np})

    @property
    def items(self) -> dict:
        return {k: v for k, v in self._items}

    @items.setter
    def items(self, items: dict):
        self._items = list(map(list, items.items()))

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._items)

    def columnCount(self, index):
        return 2

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                value = self._items[index.row()][index.column()]
                return str(value)

    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.EditRole:
            if index.column() == 1:
                try:
                    value = eval(value, self.globals)
                except Exception:
                    pass

            self._items[index.row()][index.column()] = value
            print(type(self._items[index.row()][index.column()]))
            return True
        return False

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return "Key" if section == 0 else "Value"
            else:
                return section + 1

    def flags(self, index):
        return (
            Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsEditable
        )

    def insertEmptyRow(self, index=QModelIndex()):
        self.beginInsertRows(index, len(self._items), len(self._items))
        self._items.append(["", ""])
        self.endInsertRows()
        return True

    def removeRows(self, position, nrows=1, index=QModelIndex()):
        self.beginRemoveRows(index, position, position + nrows - 1)
        self._items = self._items[:position] + self._items[position + nrows :]
        self.endRemoveRows()

        return True
