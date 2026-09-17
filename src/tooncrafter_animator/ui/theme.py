from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# WCAG-AA on #121212: body text ≥ 4.5:1, large text ≥ 3:1.
BG = "#121212"
SURFACE = "#1C1C1C"
SURFACE_2 = "#262626"
BORDER = "#3A3A3A"
TEXT = "#F2F2F2"
MUTED = "#C8C8C8"
PRIMARY = "#8AB4FF"
PRIMARY_HOVER = "#A8C5FF"
DANGER = "#FF8A80"
OK = "#80CBC4"
FOCUS = "#FFE082"
DISABLED_BG = "#2A2A2A"
DISABLED_TEXT = "#8A8A8A"


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    bg = QColor(BG)
    surface = QColor(SURFACE)
    text = QColor(TEXT)
    muted = QColor(MUTED)
    primary = QColor(PRIMARY)
    palette.setColor(QPalette.Window, bg)
    palette.setColor(QPalette.WindowText, text)
    palette.setColor(QPalette.Base, QColor(SURFACE_2))
    palette.setColor(QPalette.AlternateBase, surface)
    palette.setColor(QPalette.Text, text)
    palette.setColor(QPalette.Button, surface)
    palette.setColor(QPalette.ButtonText, text)
    palette.setColor(QPalette.Highlight, primary)
    palette.setColor(QPalette.HighlightedText, QColor("#0B0B0B"))
    palette.setColor(QPalette.ToolTipBase, surface)
    palette.setColor(QPalette.ToolTipText, text)
    palette.setColor(QPalette.PlaceholderText, muted)
    palette.setColor(QPalette.Link, primary)
    palette.setColor(QPalette.BrightText, QColor(DANGER))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(DISABLED_TEXT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(DISABLED_TEXT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(DISABLED_TEXT))
    app.setPalette(palette)
    font = QFont("Segoe UI")
    if font.family() != "Segoe UI":
        font = QFont("Inter")
    if not font.exactMatch():
        font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(_STYLESHEET)


_STYLESHEET = f"""
QWidget {{
    color: {TEXT};
    background: {BG};
    font-size: 10pt;
}}
QMainWindow, QDialog {{
    background: {BG};
}}
QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT};
}}
QLabel#hint, QLabel.hint {{
    color: {MUTED};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 8px;
    min-height: 26px;
    selection-background-color: {PRIMARY};
    selection-color: #0B0B0B;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border: 2px solid {FOCUS};
}}
QPushButton {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    min-height: 28px;
}}
QPushButton:hover {{
    background: #303030;
    border-color: {PRIMARY};
}}
QPushButton:focus {{
    border: 2px solid {FOCUS};
}}
QPushButton:disabled {{
    background: {DISABLED_BG};
    color: {DISABLED_TEXT};
    border-color: {BORDER};
}}
QPushButton#primary {{
    background: {PRIMARY};
    color: #0B0B0B;
    font-weight: 700;
    border: 1px solid {PRIMARY};
}}
QPushButton#primary:hover {{
    background: {PRIMARY_HOVER};
}}
QPushButton#primary:disabled {{
    background: {DISABLED_BG};
    color: {DISABLED_TEXT};
    border-color: {BORDER};
}}
QPushButton#danger {{
    background: {DANGER};
    color: #1A0000;
    font-weight: 700;
}}
QProgressBar {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    min-height: 18px;
    color: {TEXT};
}}
QProgressBar::chunk {{
    background: {PRIMARY};
    border-radius: 3px;
}}
QSlider::groove:horizontal {{
    height: 6px;
    background: {SURFACE_2};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: {PRIMARY};
    border: 2px solid {FOCUS};
}}
QScrollArea {{
    border: none;
}}
QSplitter::handle {{
    background: {BORDER};
    width: 2px;
}}
QToolTip {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {FOCUS};
    padding: 4px 8px;
}}
QStatusBar {{
    background: {SURFACE};
    color: {MUTED};
}}
QCheckBox:focus {{
    outline: 2px solid {FOCUS};
}}
"""
