"""Deterministic model/annotation agreement scoring; no torch/GUI dependency."""
import hashlib
from pathlib import Path
from dataset import Box, read_boxes

ALGORITHM = 'class_iou_dice_v1'

def digest(path):
    p=Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None

def image_stamp(path):
    s=Path(path).stat()
    return [s.st_size,s.st_mtime_ns]

def iou(a,b):
    inter=max(0.,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0.,min(a.y2,b.y2)-max(a.y1,b.y1))
    union=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-inter
    return inter/union if union>0 else 0.

def assignment(weights):
    """Maximum-weight one-to-one matching (Hungarian, rectangular padded with zeros)."""
    rows=len(weights);cols=len(weights[0]) if rows else 0
    if not rows or not cols:return []
    n=max(rows,cols)
    costs=[[1-(weights[i][j] if i<rows and j<cols else 0.) for j in range(n)] for i in range(n)]
    u=[0.]*(n+1);v=[0.]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
    for row in range(1,n+1):
        p[0]=row;j0=0;minimum=[float('inf')]*(n+1);used=[False]*(n+1)
        while True:
            used[j0]=True;i0=p[j0];delta=float('inf');j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cur=costs[i0-1][j-1]-u[i0]-v[j]
                    if cur<minimum[j]:minimum[j]=cur;way[j]=j0
                    if minimum[j]<delta:delta=minimum[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minimum[j]-=delta
            j0=j1
            if not p[j0]:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    return [(p[j]-1,j-1) for j in range(1,n+1) if 0<p[j]<=rows and j<=cols and weights[p[j]-1][j-1]>0]

def assess(labels,predictions,localization_iou=.5):
    """Score = 100 * 2 sum(class-consistent matched IoUs)/(Nlabels+Npredictions)."""
    preds=[Box(int(p['cls']),*p['xyxy']) for p in predictions]
    matrix=[[iou(a,b) if a.cls==b.cls else 0. for b in preds] for a in labels]
    pairs=assignment(matrix)
    matched_labels={i for i,j in pairs};matched_preds={j for i,j in pairs}
    unmatched_labels=[i for i in range(len(labels)) if i not in matched_labels]
    unmatched_preds=[j for j in range(len(preds)) if j not in matched_preds]
    label_scores=[0.]*len(labels)
    matches=[]
    for i,j in pairs:
        overlap=matrix[i][j];label_scores[i]=round(100*overlap,2)
        matches.append({'label_index':i,'prediction_index':j,'iou':overlap})
    wrong_weights=[[iou(labels[i],preds[j]) if labels[i].cls!=preds[j].cls and iou(labels[i],preds[j])>=localization_iou else 0. for j in unmatched_preds] for i in unmatched_labels]
    wrong=[{'label_index':unmatched_labels[i],'prediction_index':unmatched_preds[j],'iou':wrong_weights[i][j]} for i,j in assignment(wrong_weights)]
    wrong_labels={m['label_index'] for m in wrong};wrong_preds={m['prediction_index'] for m in wrong}
    extras=[i for i in unmatched_labels if i not in wrong_labels]
    missing=[j for j in unmatched_preds if j not in wrong_preds]
    shifted=[m['label_index'] for m in matches if m['iou']<localization_iou]
    reasons=[]
    if wrong:reasons.append(f'疑似类别不一致 {len(wrong)} 个')
    if shifted:reasons.append(f'框位置/大小偏差（IoU<{localization_iou:g}）{len(shifted)} 个')
    if extras:reasons.append(f'标签未获模型匹配 {len(extras)} 个（可能多标或模型漏检）')
    if missing:reasons.append(f'模型预测未获标签匹配 {len(missing)} 个（可能漏标或模型误检）')
    total=len(labels)+len(preds)
    score=round(100*2*sum(m['iou'] for m in matches)/total,2) if total else 100.
    if not total:reasons.append('空标签且模型未检测到目标；无法排除共同漏检')
    elif not reasons:reasons.append('类别与框匹配，按 IoU 衡量一致程度')
    return dict(score=score,label_count=len(labels),prediction_count=len(preds),label_scores=label_scores,matches=matches,
                class_mismatches=wrong,unmatched_labels=extras,unmatched_predictions=missing,shifted_labels=shifted,reasons=reasons)

def evaluate_file(label_path,predictions,names,localization_iou=.5):
    token=digest(label_path)
    if token is None:
        return dict(score=0.,reasons=['缺失标签文件'],label_scores=[],error='缺失标签',label_digest=None)
    try:
        boxes=read_boxes(Path(label_path),len(names))
        result=assess(boxes,predictions,localization_iou)
    except Exception as exc:
        result=dict(score=0.,reasons=[f'标签无法评分：{exc}'],label_scores=[],error='标签异常')
    result['label_digest']=token
    if digest(label_path)!=token:raise RuntimeError('评分期间标签被修改，请重新检查')
    return result
