"""Isolated inference process. Stdout JSON messages; never writes YOLO labels."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback
from quality import evaluate_file, image_stamp, digest, ALGORITHM

PREFIX='REVIEW_EVENT '

def emit(kind,**fields):
    print(PREFIX+json.dumps(dict(type=kind,**fields),ensure_ascii=False,allow_nan=False),flush=True)

def load_model(path):
    p=Path(path).expanduser().resolve()
    if not p.is_file() or p.suffix.lower()!='.pt':raise ValueError('请选择已有的 Ultralytics YOLO 检测模型 .pt 文件')
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO
        model=YOLO(str(p))
        if model.task!='detect':raise ValueError(f'模型任务为 {model.task}，仅支持目标检测模型')
        names=model.names
    if isinstance(names,list):names=dict(enumerate(names))
    return model,{int(k):str(v) for k,v in names.items()}

def run(request):
    model,names=load_model(request['model'])
    if request.get('mode')=='inspect':
        import torch
        emit('model',names=names,model_digest=digest(request['model']),cuda=torch.cuda.is_available())
        return
    mapping={int(k):int(v) for k,v in request['mapping'].items()}
    dataset_names=request['names']
    expected={int(k):str(v) for k,v in request['model_names'].items()}
    if names!=expected or digest(request['model'])!=request['model_digest']:raise ValueError('模型自加载后已更改，请重新加载模型')
    targets=[v for v in mapping.values() if v>=0]
    if set(mapping)!=set(names) or sorted(targets)!=list(range(len(dataset_names))):
        raise ValueError('类别映射必须一对一覆盖数据集所有类别；多余模型类别可忽略')
    params=request['params'];errors=0
    for index,sample in enumerate(request['samples']):
        try:
            before=image_stamp(sample['image'])
            # PIL RGB uses raw pixel orientation, matching QPixmap load in the editor.
            import numpy as np
            from PIL import Image
            with Image.open(sample['image']) as raw:rgb=np.asarray(raw.convert('RGB'))
            with contextlib.redirect_stdout(sys.stderr):
                result=model.predict(source=rgb[:,:,::-1].copy(),conf=params['confidence'],iou=params['nms_iou'],
                                     imgsz=params['imgsz'],device=params['device'],max_det=300,verbose=False,save=False)[0]
            predictions=[];ignored=0
            for xyxy,cls,conf in zip(result.boxes.xyxyn.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
                target=mapping[int(cls)]
                if target<0:ignored+=1;continue
                coords=[min(1.,max(0.,float(v))) for v in xyxy]
                if coords[2]>coords[0] and coords[3]>coords[1]:predictions.append(dict(cls=target,xyxy=coords,confidence=float(conf)))
            review=evaluate_file(sample['label'],predictions,dataset_names,params['localization_iou'])
            if image_stamp(sample['image'])!=before:raise RuntimeError('推理期间图片被修改，请重新检查')
            review.update(predictions=predictions,image_stamp=before,ignored_predictions=ignored)
            if len(result.boxes)>=300:
                review.update(score=None,error='预测数量达到上限',reasons=['预测达到 300 个框上限，结果可能被截断，不作质量评分'])
        except Exception as exc:
            errors+=1
            review=dict(score=None,error='推理/读取失败',reasons=[str(exc)],predictions=[],label_scores=[])
        emit('result',index=sample['index'],review=review,done=index+1,total=len(request['samples']))
    emit('complete',total=len(request['samples']),errors=errors,algorithm=ALGORITHM)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('request');args=parser.parse_args()
    try:run(json.loads(Path(args.request).read_text(encoding='utf-8')))
    except Exception as exc:
        traceback.print_exc(file=sys.stderr);emit('fatal',message=str(exc));return 1
    return 0
if __name__=='__main__':sys.exit(main())
