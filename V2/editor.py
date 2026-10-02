#!/usr/bin/env python3
"""Chinese desktop YOLO detection annotation reviewer/editor."""
import argparse
from collections import Counter
from copy import deepcopy
import csv
import os
from pathlib import Path
import sys
# cv2 can inject a conflicting Qt plugin path in inherited environments.
for key in ('QT_QPA_PLATFORM_PLUGIN_PATH','QT_QPA_FONTDIR'):
    if 'cv2' in os.environ.get(key,''):os.environ.pop(key,None)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QKeySequence, QColor
from PyQt5.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QSplitter,
    QLabel,QPushButton,QListWidget,QListWidgetItem,QLineEdit,QComboBox,QDoubleSpinBox,
    QFormLayout,QFileDialog,QMessageBox,QAction,QDialog,QDialogButtonBox,QTableWidget,
    QTableWidgetItem,QHeaderView,QCheckBox,QProgressDialog,QAbstractItemView)
from canvas import Canvas, COLORS
from dataset import Dataset, read_boxes, fingerprint

DEFAULT = Path(__file__).resolve().parents[2]/'工业电机数据集/em3_base_model_dataset_head'

class ClassDialog(QDialog):
    def __init__(self,names,parent):
        super().__init__(parent);self.setWindowTitle('类别管理 · 编号保持不变');self.resize(520,430)
        layout=QVBoxLayout(self)
        label=QLabel('重命名对整个数据集生效，保存到 data.yaml 和 classes.txt。\n已有编号不会重排或删除；更改框类别请使用右侧类别下拉框。')
        label.setWordWrap(True);layout.addWidget(label)
        self.table=QTableWidget(0,2);self.table.setHorizontalHeaderLabels(['编号','类别名称']);self.table.horizontalHeader().setSectionResizeMode(1,QHeaderView.Stretch)
        layout.addWidget(self.table)
        for name in names:self.add(name)
        button=QPushButton('＋ 新增类别');button.clicked.connect(lambda:self.add(f'class_{self.table.rowCount()}'));layout.addWidget(button)
        buttons=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def add(self,name):
        row=self.table.rowCount();self.table.insertRow(row)
        iditem=QTableWidgetItem(str(row));iditem.setFlags(Qt.ItemIsEnabled);self.table.setItem(row,0,iditem);self.table.setItem(row,1,QTableWidgetItem(name))
    def names(self):return [self.table.item(i,1).text().strip() for i in range(self.table.rowCount())]

class ReviewItem(QListWidgetItem):
    def __lt__(self, other):
        a=self.data(Qt.UserRole+1);b=other.data(Qt.UserRole+1)
        return tuple(a or [2,0,self.text()]) < tuple(b or [2,0,other.text()])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('仿生眼数据集检查与纠正 V2');self.resize(1450,900)
        self.sample_items={};self.dataset=None;self.sample=None;self.saved=[];self.token=None;self.history=[[]];self.history_index=0;self.blocked=False;self.repair=False;self.active_index=-1;self.syncing=False
        toolbar=self.addToolBar('工具');toolbar.setMovable(False)
        self.actions={}
        def action(key,title,callback,shortcut=None,checkable=False):
            a=QAction(title,self);a.setCheckable(checkable)
            if shortcut:a.setShortcut(QKeySequence(shortcut))
            a.triggered.connect(callback);toolbar.addAction(a);self.actions[key]=a;return a
        action('open','打开数据集',self.choose_dataset,'Ctrl+O')
        action('save','保存标签',self.save,'Ctrl+S');toolbar.addSeparator()
        action('prev','上一张',lambda:self.navigate(-1),'A')
        action('next','下一张',lambda:self.navigate(1),'D');toolbar.addSeparator()
        action('draw','绘制新框 [W]',self.toggle_draw,'W',True)
        action('delete','删除框',self.delete_box,'Delete')
        action('undo','撤销',self.undo,'Ctrl+Z');action('redo','重做',self.redo,'Ctrl+Shift+Z')
        action('fit','适应窗口',lambda:self.canvas.fit(),'F')
        action('classes','类别管理',self.edit_classes)
        action('scan','检查数据集',self.scan)
        action('export','导出检查报告',self.export_report)
        escape=QAction(self);escape.setShortcut('Escape');escape.triggered.connect(lambda:self.actions['draw'].setChecked(False));escape.triggered.connect(lambda:self.toggle_draw(False));self.addAction(escape)
        splitter=QSplitter();self.setCentralWidget(splitter)
        left=QWidget();ll=QVBoxLayout(left);left.setMinimumWidth(230)
        self.folder=QLabel('尚未打开数据集');self.folder.setWordWrap(True);ll.addWidget(self.folder)
        self.search=QLineEdit();self.search.setPlaceholderText('搜索图片文件名 / 相对路径');self.search.textChanged.connect(self.filter_list);ll.addWidget(self.search)
        self.filter=QComboBox();self.filter.addItems(['全部','有标注','空标签','缺失标签','格式异常','图片异常','未检查']);self.filter.currentTextChanged.connect(self.filter_list);ll.addWidget(self.filter)
        self.images=QListWidget();self.images.currentRowChanged.connect(self.choose_row);ll.addWidget(self.images)
        self.stats=QLabel();self.stats.setWordWrap(True);ll.addWidget(self.stats)
        self.autosave=QCheckBox('切图时自动保存修改');ll.addWidget(self.autosave)
        splitter.addWidget(left)
        center=QWidget();cl=QVBoxLayout(center);self.image_title=QLabel('选择数据集目录，或使用 --dataset 指定路径');cl.addWidget(self.image_title)
        self.canvas=Canvas();cl.addWidget(self.canvas,1)
        helptext=QLabel('左键：选择 / 移动框    拖动八个控制点：缩放框    滚轮：缩放图片    右键 / 中键拖动：平移\nW 绘框 · Esc 选择 · A/D 切图 · Delete 删除 · Ctrl+S 保存 · Ctrl+Z 撤销 · F 适应窗口')
        helptext.setWordWrap(True);cl.addWidget(helptext);splitter.addWidget(center)
        right=QWidget();rl=QVBoxLayout(right);right.setMinimumWidth(250)
        self.box_count=QLabel('标注框：0');rl.addWidget(self.box_count)
        self.box_list=QListWidget();self.box_list.currentRowChanged.connect(self.select_box);rl.addWidget(self.box_list)
        rl.addWidget(QLabel('选中框类别 / 新框默认类别'))
        self.class_combo=QComboBox();self.class_combo.currentIndexChanged.connect(self.change_class);rl.addWidget(self.class_combo)
        form=QFormLayout();self.spins=[]
        for name in ['左上角 X','左上角 Y','宽度 W','高度 H']:
            spin=QDoubleSpinBox();spin.setRange(0,100000);spin.setDecimals(2);spin.setKeyboardTracking(False);form.addRow(name,spin);self.spins.append(spin)
        rl.addLayout(form)
        apply=QPushButton('应用像素坐标');apply.clicked.connect(self.apply_coordinates);rl.addWidget(apply)
        names=QCheckBox('显示类别名称');names.setChecked(True);names.toggled.connect(self.set_names_visible);rl.addWidget(names)
        self.message=QLabel();self.message.setWordWrap(True);self.message.setTextInteractionFlags(Qt.TextSelectableByMouse);rl.addWidget(self.message)
        reset=QPushButton('重建异常标签…');reset.clicked.connect(self.rebuild);rl.addWidget(reset)
        reload_button=QPushButton('重新读取当前图片与标签');reload_button.clicked.connect(self.reload);rl.addWidget(reload_button)
        splitter.addWidget(right);splitter.setSizes([270,880,280])
        self.canvas.selected.connect(self.select_box);self.canvas.edited.connect(self.record);self.canvas.coordinates.connect(lambda text:self.statusBar().showMessage(text))
        self.refresh_state()

    def error(self,exc):QMessageBox.warning(self,'操作未完成',str(exc))
    def dirty(self):return self.repair or self.canvas.boxes!=self.saved
    def choose_dataset(self):
        path=QFileDialog.getExistingDirectory(self,'选择 YOLO 数据集目录',str(self.dataset.root if self.dataset else DEFAULT.parent))
        if path:self.open_dataset(path)
    def open_dataset(self,path):
        if not self.maybe_save():return
        try:
            dataset=Dataset(path)
            if not dataset.names:
                QMessageBox.information(self,'缺少类别配置','未找到 names 或 classes.txt。请按照现有标签编号顺序填写类别名称。')
                dialog=ClassDialog([],self)
                if dialog.exec_()!=QDialog.Accepted:return
                names=dialog.names()
                if not names:raise ValueError('至少需要一个类别')
                dataset.save_names(names)
        except Exception as exc:self.error(exc);return
        self.dataset=dataset;self.sample=None;self.saved=[];self.canvas.boxes=[];self.repair=False;self.active_index=-1
        self.folder.setText(str(dataset.root));self.folder.setToolTip(str(dataset.root))
        self.refresh_classes();self.search.clear();self.filter.setCurrentIndex(0)
        self.images.blockSignals(True);self.images.clear();self.sample_items={}
        for i,sample in enumerate(dataset.samples):
            item=ReviewItem(sample.relative);item.setData(Qt.UserRole,i);self.images.addItem(item);self.sample_items[i]=item
        self.images.blockSignals(False)
        self.scan();self.images.setCurrentRow(0)
    def refresh_classes(self):
        self.class_combo.blockSignals(True);self.class_combo.clear()
        if self.dataset:
            self.class_combo.addItems([f'{i} · {name}' for i,name in enumerate(self.dataset.names)]);self.canvas.names=self.dataset.names
        self.class_combo.blockSignals(False);self.canvas.class_id=max(0,self.class_combo.currentIndex())
    def scan(self):
        if not self.dataset:return
        progress=QProgressDialog('正在检查标签格式、类别与坐标…','停止',0,len(self.dataset.samples),self);progress.setWindowModality(Qt.WindowModal);progress.setMinimumDuration(300)
        for i,sample in enumerate(self.dataset.samples):
            self.dataset.inspect(sample)
            if i%100==0:
                progress.setValue(i);QApplication.processEvents()
                if progress.wasCanceled():break
        progress.close();self.update_items();self.filter_list()
    def update_items(self):
        if not self.dataset:return
        colors={'格式异常':'#ff8585','缺失标签':'#ffcf5c','空标签':'#a89cff','图片异常':'#ff8585'}
        for i,s in enumerate(self.dataset.samples):
            item=self.sample_items.get(i)
            if item:
                item.setText(f'[{s.status}] {s.relative}');item.setToolTip(str(s.label)+'\n'+s.detail);item.setForeground(QColor(colors.get(s.status,'#dce6ef')))
        counts=Counter(s.status for s in self.dataset.samples)
        self.stats.setText(f'共 {len(self.dataset.samples)} 张\n'+' · '.join(f'{k} {v}' for k,v in counts.items()))
    def filter_list(self):
        if not self.dataset:return
        term=self.search.text().strip().lower();status=self.filter.currentText()
        for i,s in enumerate(self.dataset.samples):
            item=self.sample_items.get(i)
            if item:item.setHidden(term not in s.relative.lower() or (status!='全部' and status!=s.status))
    def choose_row(self,row):
        if not self.dataset or row<0:return
        row=self.images.item(row).data(Qt.UserRole)
        if row==self.active_index:return
        if not self.maybe_save():
            self.images.blockSignals(True);self.images.setCurrentItem(self.sample_items.get(self.active_index));self.images.blockSignals(False);return
        self.load_sample(row)
    def load_sample(self,row):
        self.active_index=row;self.sample=self.dataset.samples[row];self.blocked=False;self.repair=False
        self.dataset.inspect(self.sample)
        pixmap=QPixmap(str(self.sample.image))
        boxes=[];detail=self.sample.detail
        try:
            self.token=fingerprint(self.sample.label)
        except OSError as exc:
            self.token=None;self.blocked=True;detail=f'无法读取标签：{exc}'
        if pixmap.isNull():
            self.blocked=True;self.sample.status='图片异常';detail='图片无法解码，请检查源文件。'
        elif not self.blocked:
            try:boxes=read_boxes(self.sample.label,len(self.dataset.names))
            except Exception as exc:self.blocked=True;detail=str(exc)
        self.canvas.load(pixmap,boxes,self.dataset.names);self.canvas.editable=not self.blocked
        self.saved=deepcopy(boxes);self.history=[deepcopy(boxes)];self.history_index=0
        self.message.setText(detail+'\n\n标签：'+str(self.sample.label))
        self.refresh_boxes();self.update_items();self.refresh_state()
    def navigate(self,direction):
        if not self.dataset:return
        row=self.images.currentRow()+direction
        while 0<=row<self.images.count():
            if not self.images.item(row).isHidden():self.images.setCurrentRow(row);return
            row+=direction
    def maybe_save(self):
        if not self.sample or not self.dirty():return True
        if self.autosave.isChecked():return self.save()
        answer=QMessageBox.question(self,'尚有未保存修改','是否保存当前图片的标注修改？',QMessageBox.Save|QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Save)
        if answer==QMessageBox.Cancel:return False
        return self.save() if answer==QMessageBox.Save else True
    def save(self):
        if not self.sample:return False
        if self.blocked:self.error('标签或图片异常。请先修复，或使用“重建异常标签”。');return False
        try:
            self.token=self.dataset.save_boxes(self.sample,self.canvas.boxes,self.token)
            self.saved=deepcopy(self.canvas.boxes);self.repair=False
            self.message.setText('已保存。原标签备份在数据集 .annotation_backups 目录。\n'+str(self.sample.label));self.statusBar().showMessage('标签保存成功',5000)
            self.update_items();self.filter_list();self.refresh_state();return True
        except Exception as exc:self.error(exc);return False
    def toggle_draw(self,checked):
        self.canvas.draw_mode=checked;self.canvas.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor);self.canvas.update()
    def select_box(self,index):
        if self.syncing:return
        self.syncing=True
        self.canvas.index=index;self.box_list.setCurrentRow(index)
        if 0<=index<len(self.canvas.boxes):
            b=self.canvas.boxes[index];self.class_combo.setCurrentIndex(b.cls)
            w,h=self.canvas.pixmap.width(),self.canvas.pixmap.height()
            for spin,v in zip(self.spins,[b.x1*w,b.y1*h,(b.x2-b.x1)*w,(b.y2-b.y1)*h]):spin.setValue(v)
        for spin in self.spins:spin.setEnabled(index>=0 and not self.blocked)
        self.canvas.update();self.syncing=False
    def refresh_boxes(self):
        index=self.canvas.index;self.syncing=True;self.box_list.clear()
        for i,b in enumerate(self.canvas.boxes):
            name=self.dataset.names[b.cls] if self.dataset and b.cls<len(self.dataset.names) else '?'
            item=QListWidgetItem(f'{i+1}   [{b.cls}] {name}');item.setForeground(QColor(COLORS[b.cls%len(COLORS)]));self.box_list.addItem(item)
        self.box_count.setText(f'标注框：{len(self.canvas.boxes)}');self.syncing=False
        self.select_box(index if index<len(self.canvas.boxes) else -1)
    def record(self):
        state=deepcopy(self.canvas.boxes)
        if state!=self.history[self.history_index]:
            self.history=self.history[:self.history_index+1]+[state]
            if len(self.history)>150:self.history.pop(0)
            self.history_index=len(self.history)-1
        self.refresh_boxes();self.refresh_state();self.canvas.update()
    def restore_history(self,index):
        self.history_index=index;self.canvas.boxes=deepcopy(self.history[index]);self.canvas.index=-1
        self.refresh_boxes();self.refresh_state();self.canvas.update()
    def undo(self):
        if self.history_index>0:self.restore_history(self.history_index-1)
    def redo(self):
        if self.history_index+1<len(self.history):self.restore_history(self.history_index+1)
    def delete_box(self):
        i=self.canvas.index
        if not self.blocked and 0<=i<len(self.canvas.boxes):self.canvas.boxes.pop(i);self.canvas.index=-1;self.record()
    def change_class(self,index):
        if index<0:return
        self.canvas.class_id=index
        if self.syncing or self.blocked:return
        i=self.canvas.index
        if 0<=i<len(self.canvas.boxes) and self.canvas.boxes[i].cls!=index:self.canvas.boxes[i].cls=index;self.record()
    def apply_coordinates(self):
        i=self.canvas.index
        if self.blocked or not 0<=i<len(self.canvas.boxes):return
        x,y,w,h=[spin.value() for spin in self.spins];iw,ih=self.canvas.pixmap.width(),self.canvas.pixmap.height()
        b=deepcopy(self.canvas.boxes[i]);b.x1=x/iw;b.y1=y/ih;b.x2=(x+w)/iw;b.y2=(y+h)/ih
        try:b.validate(len(self.dataset.names))
        except ValueError as exc:self.error(exc);return
        self.canvas.boxes[i]=b;self.record()
    def refresh_state(self):
        dirty=self.dirty();name=self.sample.relative if self.sample else '未打开数据集'
        self.setWindowTitle(f'{"* " if dirty else ""}{name} — 仿生眼数据集检查与纠正 V2')
        if self.sample:
            self.image_title.setText(f'{self.active_index+1}/{len(self.dataset.samples)}   {name}   {self.canvas.pixmap.width()} × {self.canvas.pixmap.height()}'+('   ● 未保存' if dirty else ''))
        self.actions['undo'].setEnabled(self.history_index>0);self.actions['redo'].setEnabled(self.history_index+1<len(self.history))
        self.actions['save'].setEnabled(bool(self.sample) and not self.blocked)
    def edit_classes(self):
        if not self.dataset:return
        dialog=ClassDialog(self.dataset.names,self)
        if dialog.exec_()!=QDialog.Accepted:return
        try:self.dataset.save_names(dialog.names())
        except Exception as exc:self.error(exc);return
        selected=self.canvas.index;self.refresh_classes();self.refresh_boxes();self.select_box(selected);self.canvas.update()
        self.statusBar().showMessage('类别名称已保存，已有类别编号保持不变',5000)
    def set_names_visible(self,checked):self.canvas.show_names=checked;self.canvas.update()
    def rebuild(self):
        if not self.sample or not self.blocked or self.canvas.pixmap.isNull():return
        result=QMessageBox.warning(self,'重建异常标签','原标签无法完整读取。重建会从零个框开始，保存时备份原文件后覆盖。是否继续？',QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if result!=QMessageBox.Yes:return
        self.blocked=False;self.canvas.editable=True;self.repair=True;self.message.setText('重建模式：请重新绘框并保存。原标签将在保存时备份。');self.refresh_state()
    def reload(self):
        if not self.sample:return
        if self.dirty():
            answer=QMessageBox.question(self,'重新读取','丢弃当前未保存修改，并从磁盘重新读取？',QMessageBox.Discard|QMessageBox.Cancel,QMessageBox.Cancel)
            if answer!=QMessageBox.Discard:return
        self.load_sample(self.active_index)
    def export_report(self):
        if not self.dataset:return
        path,_=QFileDialog.getSaveFileName(self,'导出标签检查报告',str(self.dataset.root/'annotation_report.csv'),'CSV (*.csv)')
        if not path:return
        try:
            with open(path,'w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(['图片','标签','检查状态','详情'])
                for s in self.dataset.samples:writer.writerow([str(s.image),str(s.label),s.status,s.detail])
            self.statusBar().showMessage('报告已导出：'+path,5000)
        except Exception as exc:self.error(exc)
    def closeEvent(self,event):
        if self.maybe_save():event.accept()
        else:event.ignore()

STYLE='''
QWidget { background: #1b2632; color: #dce6ef; font-size: 13px; }
QToolBar { spacing: 7px; padding: 7px; border-bottom: 1px solid #344658; }
QToolButton, QPushButton { background: #2c4052; padding: 7px 10px; border: 1px solid #405970; border-radius: 4px; }
QToolButton:hover, QPushButton:hover { background: #365c77; }
QToolButton:checked { background: #126f80; }
QLineEdit,QComboBox,QDoubleSpinBox { background: #121d28; padding: 6px; border: 1px solid #405064; border-radius: 3px; }
QListWidget,QTableWidget { background: #14202b; border: 1px solid #344658; }
QListWidget::item { padding: 5px; }
QListWidget::item:selected { background: #23576c; }
QLabel { background: transparent; }
QStatusBar { background: #12202b; }
QWidget:disabled { color: #697986; }
'''

def main():
    parser=argparse.ArgumentParser(description='仿生眼数据集检查与纠正 V2')
    parser.add_argument('--dataset',type=Path,help='数据集目录或 data.yaml；默认打开 head 数据集')
    args=parser.parse_args()
    app=QApplication(sys.argv[:1]);app.setApplicationName('仿生眼数据集检查与纠正 V2');app.setStyle('Fusion');app.setStyleSheet(STYLE)
    window=MainWindow();window.show()
    path=args.dataset or (DEFAULT if DEFAULT.exists() else None)
    if path:QTimer.singleShot(0,lambda:window.open_dataset(path))
    sys.exit(app.exec_())
if __name__=='__main__':main()
