#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
from editor import STYLE,DEFAULT
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from review_ui import ReviewWindow as MainWindow

def main():
    parser=argparse.ArgumentParser(description='仿生眼数据集检查与纠正 V2 · 模型辅助质量检查')
    parser.add_argument('--dataset',type=Path)
    args=parser.parse_args()
    app=QApplication(sys.argv[:1]);app.setApplicationName('仿生眼数据集检查与纠正 V2');app.setStyle('Fusion');app.setStyleSheet(STYLE)
    window=MainWindow();window.show();path=args.dataset or (DEFAULT if DEFAULT.exists() else None)
    if path:QTimer.singleShot(0,lambda:window.open_dataset(path))
    sys.exit(app.exec_())
if __name__=='__main__':main()
