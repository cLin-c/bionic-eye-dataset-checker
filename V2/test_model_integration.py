"""Optional real-model test. Copies a few images/labels into a temporary dataset."""
import argparse
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import shutil
import tempfile
import time
import json
import yaml
from PyQt5.QtWidgets import QApplication
from app import MainWindow,STYLE,DEFAULT
from dataset import Dataset
from quality import digest

def wait(window,app,seconds=90):
    deadline=time.monotonic()+seconds
    while window.process is not None and time.monotonic()<deadline:
        app.processEvents();time.sleep(.02)
    if window.process is not None:
        window.stop_review()
        raise TimeoutError('模型任务超时')
    app.processEvents()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--python',required=True);args=parser.parse_args()
    source=Dataset(DEFAULT);selected=[next(s for s in source.samples if s.image.stem==name) for name in ['head_1559','head_3410','head_4500']]
    tokens={s.label:digest(s.label) for s in selected}
    app=QApplication([]);app.setStyle('Fusion');app.setStyleSheet(STYLE)
    with tempfile.TemporaryDirectory(prefix='bionic-v2-integration-') as temp:
        root=Path(temp);(root/'images/test').mkdir(parents=True);(root/'labels/test').mkdir(parents=True)
        for s in selected:
            shutil.copy2(s.image,root/'images/test'/s.image.name);shutil.copy2(s.label,root/'labels/test'/s.label.name)
        (root/'data.yaml').write_text(yaml.safe_dump(dict(names=dict(enumerate(source.names)),nc=len(source.names))),encoding='utf-8')
        window=MainWindow();errors=[];window.error=lambda e:errors.append(str(e));window.show();window.open_dataset(root);window.python_path.setText(args.python)
        window.load_model(args.model);wait(window,app)
        assert window.model_info,errors
        assert window.validate_mapping(),window.mapping
        window.device.setCurrentIndex(0);window.start_review();wait(window,app)
        assert not errors,errors
        assert len(window.reviews)==3
        assert all(r.get('score') is not None for r in window.reviews.values()),window.reviews
        report={window.dataset.samples[i].relative:dict(score=r['score'],predictions=len(r['predictions']),reasons=r['reasons']) for i,r in window.reviews.items()}
        window.jump_low();app.processEvents();window.grab().save(str(Path(__file__).with_name('V2界面预览.png')))
        current=window.active_index
        window.start_review(current=True);wait(window,app)
        assert not errors,errors
        assert len(window.reviews)==3
        # Cancellation keeps GUI alive and never writes labels.
        window.start_review();window.stop_review();wait(window,app)
        assert not errors,errors
        assert all(digest(s.label)==tokens[s.label] for s in selected)
        window.close()
        print(json.dumps({'real_model':args.model,'results':report,'original_labels_unchanged':True,'current_rescan':True,'cancel':True},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
