"""V2 model-assisted review UI; inference is isolated in a QProcess."""
from copy import deepcopy
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from PyQt5.QtCore import Qt,QProcess,QTimer
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QLineEdit,
    QDoubleSpinBox,QSpinBox,QComboBox,QCheckBox,QDockWidget,QProgressBar,QFileDialog,
    QDialog,QTableWidget,QTableWidgetItem,QDialogButtonBox,QHeaderView,QMessageBox,QPlainTextEdit)
from editor import MainWindow as Editor
from dataset import atomic_write
from quality import evaluate_file,digest,image_stamp,ALGORITHM

class MappingDialog(QDialog):
    def __init__(self,model_names,dataset_names,mapping,parent):
        super().__init__(parent);self.setWindowTitle('模型类别 → 数据集类别');self.resize(670,440)
        layout=QVBoxLayout(self);tip=QLabel('按名称自动对应，可手动修正。每个数据集类别必须对应一个模型类别。\n不参与检查的模型类别选“忽略”。映射仅用于评分，不改写模型或标签。');layout.addWidget(tip)
        self.table=QTableWidget(len(model_names),2);self.table.setHorizontalHeaderLabels(['模型类别','数据集类别']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch);layout.addWidget(self.table)
        self.combos={}
        for row,(key,name) in enumerate(sorted(model_names.items())):
            item=QTableWidgetItem(f'{key}: {name}');item.setFlags(Qt.ItemIsEnabled);self.table.setItem(row,0,item)
            combo=QComboBox();combo.addItem('忽略此模型类别',-1)
            for i,n in enumerate(dataset_names):combo.addItem(f'{i}: {n}',i)
            combo.setCurrentIndex(combo.findData(mapping.get(key,-1)));self.table.setCellWidget(row,1,combo);self.combos[key]=combo
        buttons=QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel);buttons.accepted.connect(self.accept);buttons.rejected.connect(self.reject);layout.addWidget(buttons)
    def mapping(self):return {key:combo.currentData() for key,combo in self.combos.items()}

class ReviewWindow(Editor):
    def __init__(self):
        self.reviews={};self.model_info=None;self.mapping={};self.process=None;self.context=None;self.mode=None
        self.temp_job=None;self.pending_context=None;self.output_buffer=b'';self.worker_log='';self.cancelled=False;self.saw_complete=False
        super().__init__()
        self.resize(1510,1030)
        left=self.centralWidget().widget(0).layout()
        self.quality_threshold=QDoubleSpinBox();self.quality_threshold.setRange(0,100);self.quality_threshold.setValue(80);self.quality_threshold.setSuffix(' 分');self.quality_threshold.setToolTip('严格低于此分数的图片优先复查；调节后无需重新推理')
        line=QHBoxLayout();line.addWidget(QLabel('质量阈值'));line.addWidget(self.quality_threshold);left.insertLayout(3,line)
        self.low_only=QCheckBox('仅显示低于阈值 / 检查异常');left.insertWidget(4,self.low_only)
        self.low_first=QCheckBox('按质量从低到高排序');self.low_first.setChecked(True);left.insertWidget(5,self.low_first)
        self.review_stats=QLabel('模型尚未检查');self.review_stats.setWordWrap(True);left.insertWidget(6,self.review_stats)
        jump=QPushButton('跳到首个待复查图片');jump.clicked.connect(self.jump_low);left.insertWidget(7,jump)
        self.quality_threshold.valueChanged.connect(lambda:self.refresh_reviews());self.low_only.toggled.connect(self.apply_quality_filter);self.low_first.toggled.connect(self.sort_reviews)
        dock=QDockWidget('模型辅助检查 · 分数表示与模型的一致程度，需人工确认',self);dock.setAllowedAreas(Qt.BottomDockWidgetArea|Qt.TopDockWidgetArea)
        panel=QWidget();layout=QVBoxLayout(panel);dock.setWidget(panel);self.addDockWidget(Qt.BottomDockWidgetArea,dock)
        row=QHBoxLayout();layout.addLayout(row)
        self.model_path=QLineEdit();self.model_path.setReadOnly(True);self.model_path.setPlaceholderText('加载训练好的 YOLO 检测模型 .pt')
        self.load_button=QPushButton('加载模型…');self.load_button.clicked.connect(self.choose_model);row.addWidget(self.load_button);row.addWidget(self.model_path,2)
        row.addWidget(QLabel('推理 Python'))
        candidates=[Path('/home/ccl/miniconda3/envs/yolov11/bin/python'),Path(sys.executable)]
        self.python_path=QLineEdit(str(next(p for p in candidates if p.exists())));self.python_path.setMinimumWidth(260);row.addWidget(self.python_path,1)
        self.python_button=QPushButton('选择…');self.python_button.clicked.connect(self.choose_python);row.addWidget(self.python_button)
        self.mapping_button=QPushButton('类别映射…');self.mapping_button.clicked.connect(self.edit_mapping);row.addWidget(self.mapping_button)
        row=QHBoxLayout();layout.addLayout(row)
        self.confidence=QDoubleSpinBox();self.confidence.setRange(.01,.99);self.confidence.setSingleStep(.05);self.confidence.setValue(.25)
        self.iou_threshold=QDoubleSpinBox();self.iou_threshold.setRange(.05,.95);self.iou_threshold.setSingleStep(.05);self.iou_threshold.setValue(.5)
        self.nms=QDoubleSpinBox();self.nms.setRange(.05,.95);self.nms.setSingleStep(.05);self.nms.setValue(.7)
        self.imgsz=QSpinBox();self.imgsz.setRange(64,2048);self.imgsz.setSingleStep(32);self.imgsz.setValue(640)
        self.device=QComboBox();self.device.addItem('CPU','cpu');self.device.addItem('GPU 0','0')
        for label,widget in [('预测置信度',self.confidence),('偏差提示 IoU',self.iou_threshold),('NMS IoU',self.nms),('图像尺寸',self.imgsz),('设备',self.device)]:row.addWidget(QLabel(label));row.addWidget(widget)
        self.overlay=QCheckBox('叠加模型预测（黄色虚线）');self.overlay.setChecked(True);self.overlay.toggled.connect(self.refresh_review_detail);row.addWidget(self.overlay)
        row.addStretch()
        row=QHBoxLayout();layout.addLayout(row)
        self.start_button=QPushButton('检查全部图片');self.start_button.clicked.connect(lambda:self.start_review(False));row.addWidget(self.start_button)
        self.current_button=QPushButton('复查当前图片');self.current_button.clicked.connect(lambda:self.start_review(True));row.addWidget(self.current_button)
        self.stop_button=QPushButton('停止检查');self.stop_button.clicked.connect(self.stop_review);self.stop_button.setEnabled(False);row.addWidget(self.stop_button)
        self.import_button=QPushButton('载入评分结果');self.import_button.clicked.connect(self.import_results);row.addWidget(self.import_button)
        self.report_button=QPushButton('导出质量 CSV');self.report_button.clicked.connect(self.export_quality);row.addWidget(self.report_button)
        self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0);row.addWidget(self.progress,1)
        self.model_status=QLabel('请选择与数据集对应的模型。');layout.addWidget(self.model_status)
        self.review_detail=QPlainTextEdit();self.review_detail.setReadOnly(True);self.review_detail.setMaximumHeight(94);self.review_detail.setPlaceholderText('选中图片后显示质量分数、每个标签框的得分及疑似问题。');layout.addWidget(self.review_detail)
        self.refresh_timer=QTimer(self);self.refresh_timer.setSingleShot(True);self.refresh_timer.setInterval(400);self.refresh_timer.timeout.connect(self.refresh_reviews)
        for widget in [self.confidence,self.iou_threshold,self.nms,self.imgsz]:widget.valueChanged.connect(self.invalidate_parameters)
        self.device.currentIndexChanged.connect(self.invalidate_parameters);self.python_path.textEdited.connect(self.invalidate_parameters)

    def params(self):
        return dict(confidence=self.confidence.value(),localization_iou=self.iou_threshold.value(),nms_iou=self.nms.value(),imgsz=self.imgsz.value(),device=self.device.currentData())
    def make_context(self):
        return dict(algorithm=ALGORITHM,model=self.model_path.text(),model_digest=self.model_info['model_digest'],
                    model_names={str(k):v for k,v in self.model_info['names'].items()},mapping={str(k):v for k,v in self.mapping.items()},
                    names=list(self.dataset.names),root=str(self.dataset.root),params=self.params(),python=self.python_path.text())
    def invalidate_parameters(self,*args):
        if self.reviews:
            self.reviews={};self.context=None;self.refresh_reviews();self.model_status.setText('检查参数已更改，旧评分已清除，请重新检查。')
    def open_dataset(self,path):
        if self.process:
            self.error('请先停止当前模型任务，再切换数据集。');return
        old=self.dataset;previous_reviews=self.reviews;previous_context=self.context
        self.reviews={};self.context=None
        super().open_dataset(path)
        if self.dataset is old:
            self.reviews=previous_reviews;self.context=previous_context;self.refresh_reviews()
        else:
            self.reviews={};self.context=None
            if self.model_info:self.auto_mapping()
            self.refresh_reviews()
    def choose_python(self):
        path,_=QFileDialog.getOpenFileName(self,'选择具有 torch 和 ultralytics 的 Python 解释器',self.python_path.text())
        if path:self.python_path.setText(path);self.invalidate_parameters()
    def choose_model(self):
        if not self.dataset:self.error('请先打开数据集。');return
        path,_=QFileDialog.getOpenFileName(self,'加载已训练 YOLO 检测模型',str(Path(__file__).resolve().parents[2]),'YOLO 检测模型 (*.pt)')
        if path:self.load_model(path)
    def load_model(self,path):
        if self.process:return
        self.model_info=None;self.mapping={};self.reviews={};self.context=None;self.model_path.setText(str(Path(path).resolve()));self.refresh_reviews()
        self.launch(dict(mode='inspect',model=self.model_path.text()),'inspect')
    def auto_mapping(self):
        names=self.dataset.names
        self.mapping={int(k):(names.index(v) if v in names else -1) for k,v in self.model_info['names'].items()}
    def validate_mapping(self):
        mapped=[v for v in self.mapping.values() if v>=0]
        return sorted(mapped)==list(range(len(self.dataset.names))) and set(self.mapping)==set(self.model_info['names'])
    def edit_mapping(self):
        if not self.dataset or not self.model_info:self.error('请先打开数据集并加载模型。');return
        dialog=MappingDialog(self.model_info['names'],self.dataset.names,self.mapping,self)
        if dialog.exec_()==QDialog.Accepted:
            previous=self.mapping;self.mapping=dialog.mapping()
            if not self.validate_mapping():self.mapping=previous;self.error('映射必须一对一覆盖全部数据集类别。');return
            if self.mapping!=previous:self.invalidate_parameters()
            self.model_status.setText('类别映射已设置：'+self.mapping_summary())
    def mapping_summary(self):
        return '；'.join(f'{k}:{self.model_info["names"][k]} → {v}:{self.dataset.names[v]}' for k,v in self.mapping.items() if v>=0)
    def start_review(self,current=False):
        if self.process:return
        if not self.dataset or not self.model_info:self.error('请先打开数据集并加载模型。');return
        if not self.validate_mapping():self.edit_mapping()
        if not self.validate_mapping():return
        # The worker evaluates disk labels. Persist current edits before taking a snapshot.
        if self.dirty() and not self.save():return
        context=self.make_context()
        if self.context!=context or not current:self.reviews={}
        self.context=context
        indices=[self.active_index] if current and self.sample else list(range(len(self.dataset.samples)))
        samples=[dict(index=i,image=str(self.dataset.samples[i].image),label=str(self.dataset.samples[i].label)) for i in indices]
        request=dict(context,mode='review',samples=samples)
        self.pending_context=deepcopy(context)
        self.progress.setRange(0,len(samples));self.progress.setValue(0)
        self.launch(request,'review')
    def launch(self,request,mode):
        executable=self.python_path.text().strip()
        if not Path(executable).is_file():self.error('推理 Python 路径不存在，请选择正确的解释器。');return
        self.temp_job=tempfile.TemporaryDirectory(prefix='bionic-review-');request_path=Path(self.temp_job.name)/'request.json'
        request_path.write_text(json.dumps(request,ensure_ascii=False),encoding='utf-8')
        self.mode=mode;self.cancelled=False;self.saw_complete=False;self.worker_log='';self.output_buffer=b''
        process=QProcess(self);self.process=process
        process.setProgram(executable);process.setArguments(['-u',str(Path(__file__).with_name('model_worker.py')),str(request_path)])
        process.setWorkingDirectory(str(Path(__file__).parent))
        process.readyReadStandardOutput.connect(self.read_output);process.readyReadStandardError.connect(self.read_errors)
        process.finished.connect(self.worker_finished);process.errorOccurred.connect(self.process_error)
        self.set_busy(True);self.model_status.setText('正在加载模型…' if mode=='inspect' else '模型检查运行中，可继续浏览和编辑；评分会核对最新保存的标签。')
        if mode=='inspect':self.progress.setRange(0,0)
        process.start()
    def set_busy(self,busy):
        for w in [self.load_button,self.python_path,self.python_button,self.mapping_button,self.start_button,self.current_button,self.confidence,self.iou_threshold,self.nms,self.imgsz,self.device,self.import_button]:w.setEnabled(not busy)
        self.actions['open'].setEnabled(not busy);self.actions['classes'].setEnabled(not busy);self.stop_button.setEnabled(busy)
    def read_errors(self):
        if self.process:self.worker_log=(self.worker_log+bytes(self.process.readAllStandardError()).decode('utf-8','replace'))[-16000:]
    def read_output(self):
        if not self.process:return
        self.output_buffer+=bytes(self.process.readAllStandardOutput())
        while b'\n' in self.output_buffer:
            line,self.output_buffer=self.output_buffer.split(b'\n',1)
            if not line.startswith(b'REVIEW_EVENT '):continue
            try:self.handle_event(json.loads(line[len(b'REVIEW_EVENT '):]))
            except Exception as exc:self.worker_log+='\n解析检查结果失败：'+str(exc)
    def handle_event(self,event):
        kind=event['type']
        if kind=='model':
            event['names']={int(k):v for k,v in event['names'].items()};self.model_info=event;self.auto_mapping();self.saw_complete=True
            self.device.setCurrentIndex(1 if event.get('cuda') else 0)
            self.model_status.setText('模型已加载。'+(self.mapping_summary() if self.validate_mapping() else '类别名称不完全对应，请设置类别映射。'))
        elif kind=='result':
            i=int(event['index']);review=event['review'];self.reviews[i]=review
            self.ensure_fresh(i)
            self.progress.setValue(event['done'])
            if not self.refresh_timer.isActive():self.refresh_timer.start()
        elif kind=='fatal':self.worker_log+='\n'+event['message']
        elif kind=='complete':self.saw_complete=True
    def process_error(self,error):
        if error==QProcess.FailedToStart:
            self.worker_log+='\n无法启动推理 Python：'+self.process.errorString();self.worker_finished(-1,QProcess.CrashExit)
    def worker_finished(self,exit_code,exit_status):
        if not self.process:return
        self.read_output();self.read_errors();mode=self.mode
        process=self.process;self.process=None;process.deleteLater()
        if self.temp_job:self.temp_job.cleanup();self.temp_job=None
        self.set_busy(False);self.refresh_timer.stop();self.refresh_reviews();self.sort_reviews()
        if mode=='review':
            self.persist_results()
            self.model_status.setText(f'{"已停止，保留已完成结果" if self.cancelled else "检查完成"}：已获得 {len(self.reviews)} 张评分。可调整左侧质量阈值并优先纠正。')
        else:
            self.progress.setRange(0,1);self.progress.setValue(1 if self.model_info else 0)
            if self.cancelled:self.model_status.setText('模型加载已停止。')
        if not self.cancelled and (exit_code!=0 or not self.saw_complete):
            self.model_status.setText('模型任务未成功完成。已完成的评分仍保留。')
            self.error((self.worker_log[-4000:] or '推理进程未返回完整结果')+'\n请确认所选 Python 环境安装了兼容模型的 torch 和 ultralytics。')
    def stop_review(self):
        if not self.process:return
        self.cancelled=True;process=self.process;process.terminate();self.stop_button.setEnabled(False)
        QTimer.singleShot(2000,lambda:self.kill_if_running(process))
    def kill_if_running(self,process):
        if self.process is process and process.state()!=QProcess.NotRunning:process.kill()
    def ensure_fresh(self,i):
        review=self.reviews.get(i)
        if not review or 'image_stamp' not in review:return
        sample=self.dataset.samples[i]
        try:
            if image_stamp(sample.image)!=review['image_stamp']:
                review.update(score=None,error='图片已变化',stale=True,reasons=['图片已变化，请重新推理']);return
            if digest(sample.label)!=review.get('label_digest'):
                prediction=review['predictions'];stamp=review['image_stamp']
                updated=evaluate_file(sample.label,prediction,self.dataset.names,(self.context or self.pending_context)['params']['localization_iou'])
                updated.update(predictions=prediction,image_stamp=stamp,ignored_predictions=review.get('ignored_predictions',0))
                self.reviews[i]=updated
        except Exception as exc:review.update(score=None,error='读取失败',stale=True,reasons=[str(exc)])
    def load_sample(self,row):
        super().load_sample(row)
        # Auto-save may reorder the list while a navigation signal is being handled.
        self.images.blockSignals(True)
        self.images.setCurrentItem(self.sample_items.get(row))
        self.images.blockSignals(False)
        if hasattr(self,'review_detail'):
            self.ensure_fresh(row);self.update_items();self.refresh_review_detail()
    def save(self):
        ok=super().save()
        if ok and hasattr(self,'review_detail'):
            self.ensure_fresh(self.active_index);self.refresh_reviews();self.sort_reviews()
            if not self.process:self.persist_results()
        return ok
    def edit_classes(self):
        before=list(self.dataset.names) if self.dataset else []
        super().edit_classes()
        if self.dataset and self.dataset.names!=before:
            self.reviews={};self.context=None
            if self.model_info:self.auto_mapping()
            self.refresh_reviews();self.model_status.setText('类别名称已改变，请确认模型类别映射并重新检查。')
    def refresh_state(self):
        super().refresh_state()
        if hasattr(self,'review_detail'):self.refresh_review_detail()
    def refresh_reviews(self):
        self.update_items();self.apply_quality_filter();self.refresh_review_detail()
    def update_items(self):
        super().update_items()
        if not self.dataset:return
        for i,review in self.reviews.items():
            item=self.sample_items.get(i)
            if item is None:continue
            score=review.get('score');sample=self.dataset.samples[i]
            text=f'{score:.1f}分' if score is not None else '检查异常'
            item.setText(f'[{text}] [{sample.status}] {sample.relative}')
            item.setToolTip(str(sample.label)+'\n'+'\n'.join(review.get('reasons',[])))
            if score is None or score<self.quality_threshold.value():item.setForeground(QColor('#ffad80'))
        for i,item in self.sample_items.items():
            review=self.reviews.get(i)
            score=review.get('score') if review else None
            item.setData(Qt.UserRole+1, [0,-1,self.dataset.samples[i].relative] if review and score is None else ([1,score,self.dataset.samples[i].relative] if review else [2,0,self.dataset.samples[i].relative]))
    def filter_list(self):
        super().filter_list()
        if hasattr(self,'low_only') and self.low_only.isChecked():
            for i,item in self.sample_items.items():
                review=self.reviews.get(i)
                if not review or (review.get('score') is not None and review['score']>=self.quality_threshold.value()):item.setHidden(True)
    def apply_quality_filter(self,*args):
        self.filter_list()
        if not hasattr(self,'review_stats'):return
        bad=sum(r.get('score') is None or r['score']<self.quality_threshold.value() for r in self.reviews.values())
        self.review_stats.setText(f'模型已检查 {len(self.reviews)} 张 · 待复查 {bad} 张\n质量 < {self.quality_threshold.value():g} 分及检查异常')
    def jump_low(self):
        self.sort_reviews()
        for row in range(self.images.count()):
            item=self.images.item(row);review=self.reviews.get(item.data(Qt.UserRole))
            if not item.isHidden() and review and (review.get('score') is None or review['score']<self.quality_threshold.value()):
                self.images.setCurrentRow(row);self.images.scrollToItem(item);return
        self.statusBar().showMessage('当前筛选范围内没有低于阈值或检查异常的图片。',5000)
    def sort_reviews(self,*args):
        if not self.dataset:return
        self.update_items()
        self.images.blockSignals(True)
        if self.low_first.isChecked():self.images.sortItems(Qt.AscendingOrder)
        else:
            for i,item in self.sample_items.items():item.setData(Qt.UserRole+1,[0,i,''])
            self.images.sortItems(Qt.AscendingOrder)
        if self.active_index in self.sample_items:self.images.setCurrentItem(self.sample_items[self.active_index])
        self.images.blockSignals(False)
    def refresh_review_detail(self,*args):
        if not hasattr(self,'review_detail'):return
        review=self.reviews.get(self.active_index)
        self.canvas.predictions=(review.get('predictions',[]) if review and not review.get('stale') else [])
        self.canvas.show_predictions=self.overlay.isChecked();self.canvas.update()
        if not review:self.review_detail.setPlainText('当前图片尚未进行模型检查。');return
        score=review.get('score');title=f'质量分数：{score:.2f}/100' if score is not None else '当前图片检查异常，需优先复查'
        if self.dirty():title+='   【当前标注有未保存修改，分数基于上次保存；保存后自动重新比较】'
        details='；'.join(f'框 {i+1}: {v:.1f} 分' for i,v in enumerate(review.get('label_scores',[])))
        self.review_detail.setPlainText(title+'\n'+'；'.join(review.get('reasons',[]))+'\n'+details)
    def persist_results(self):
        if not self.dataset or not self.context:return
        document=dict(version=2,context=self.context,updated=datetime.now().isoformat(),results={self.dataset.samples[i].relative:r for i,r in self.reviews.items()})
        try:atomic_write(self.dataset.root/'.model_review/latest.json',json.dumps(document,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8'))
        except Exception as exc:self.error('评分结果缓存保存失败（原标签未受影响）：'+str(exc))
    def import_results(self):
        if not self.dataset or not self.model_info:self.error('请先打开数据集、加载同一模型并设置类别映射。');return
        path,_=QFileDialog.getOpenFileName(self,'载入评分结果',str(self.dataset.root/'.model_review/latest.json'),'JSON (*.json)')
        if not path:return
        try:
            document=json.loads(Path(path).read_text(encoding='utf-8'));context=self.make_context()
            # The Python path may change without altering inference settings.
            stored=dict(document['context']);stored.pop('python',None);compare=dict(context);compare.pop('python',None)
            if stored!=compare:raise ValueError('结果的模型、数据集、类别映射或推理参数与当前设置不一致，请按原设置加载或重新检查。')
            self.context=context;self.reviews={i:document['results'][s.relative] for i,s in enumerate(self.dataset.samples) if s.relative in document['results']}
            for i in list(self.reviews):self.ensure_fresh(i)
            self.refresh_reviews();self.sort_reviews();self.model_status.setText(f'已载入 {len(self.reviews)} 张评分，并核对标签与图片是否变化。')
        except Exception as exc:self.error(exc)
    def export_quality(self):
        if not self.dataset or not self.reviews:self.error('尚无模型检查结果。');return
        path,_=QFileDialog.getSaveFileName(self,'导出质量报告',str(self.dataset.root/'model_quality_report.csv'),'CSV (*.csv)')
        if not path:return
        try:
            for i in list(self.reviews):self.ensure_fresh(i)
            with open(path,'w',encoding='utf-8-sig',newline='') as f:
                writer=csv.writer(f);writer.writerow(['图片','标签','质量分数','需复查','质量阈值','疑似问题','框得分','模型','推理参数','类别映射'])
                for i,r in sorted(self.reviews.items(),key=lambda kv:kv[1].get('score') if kv[1].get('score') is not None else -1):
                    s=self.dataset.samples[i];score=r.get('score');writer.writerow([str(s.image),str(s.label),score,score is None or score<self.quality_threshold.value(),self.quality_threshold.value(),'；'.join(r.get('reasons',[])),r.get('label_scores',[]),self.context['model'],json.dumps(self.context['params']),json.dumps(self.context['mapping'])])
            self.model_status.setText('质量报告已导出：'+path)
        except Exception as exc:self.error(exc)
    def closeEvent(self,event):
        if self.process:
            self.error('模型检查正在运行，请先点击“停止检查”，任务停止后再关闭。');event.ignore();return
        super().closeEvent(event)
