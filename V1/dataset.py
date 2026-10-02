"""YOLO detection dataset I/O, validation and recoverable writes."""
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import math
import os
import shutil
import tempfile
import yaml

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}

@dataclass(eq=True)
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def validate(self, count):
        vals = (self.x1, self.y1, self.x2, self.y2)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError('坐标包含非有限数值')
        if not 0 <= self.cls < count:
            raise ValueError(f'未知类别编号 {self.cls}')
        if not (0 <= self.x1 < self.x2 <= 1 and 0 <= self.y1 < self.y2 <= 1):
            raise ValueError('框坐标越界或宽高为零')

@dataclass
class Sample:
    image: Path
    label: Path
    relative: str
    status: str = '未检查'
    detail: str = ''


def read_boxes(path, count):
    """Never silently drop malformed lines. Missing labels are separately tracked."""
    if not path.exists():
        return []
    result = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError('需要 5 列：class x_center y_center width height；不支持分割/姿态标签')
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            if not all(math.isfinite(v) for v in (x,y,w,h)) or w <= 0 or h <= 0:
                raise ValueError('宽高必须大于零，坐标必须为有限数值')
            coords = (x-w/2,y-h/2,x+w/2,y+h/2)
            # YOLO six-decimal export can overshoot a boundary by half a unit.
            if min(coords) < -1e-6 or max(coords) > 1+1e-6:
                raise ValueError('标注框超出图像边界')
            box = Box(cls, *(min(1.,max(0.,v)) for v in coords))
            box.validate(count)
            result.append(box)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f'{path.name} 第 {number} 行：{exc}') from exc
    return result


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def fingerprint(path):
    return path.read_bytes() if path.exists() else None


class Dataset:
    def __init__(self, root):
        selected = Path(root).expanduser().resolve()
        self.yaml_path = selected if selected.is_file() else None
        self.root = selected.parent if self.yaml_path else selected
        if self.root.name == 'images' and (self.root.parent/'labels').is_dir():
            self.root = self.root.parent
        if self.yaml_path is None:
            self.yaml_path = next((self.root/n for n in ('data.yaml','dataset.yaml','data.yml') if (self.root/n).is_file()), self.root/'data.yaml')
        self.config = yaml.safe_load(self.yaml_path.read_text(encoding='utf-8-sig')) if self.yaml_path.exists() else {}
        self.config = self.config or {}
        if not isinstance(self.config, dict):
            raise ValueError('data.yaml 须为键值配置')
        # Prefer the selected dataset folder: exported data.yaml can have stale paths.
        if not (self.root/'images').is_dir() and self.config.get('path'):
            candidate = Path(str(self.config['path'])).expanduser()
            if not candidate.is_absolute():
                candidate = self.root/candidate
            if (candidate/'images').is_dir():
                self.root = candidate.resolve()
        raw = self.config.get('names')
        classes = self.root/'classes.txt'
        if isinstance(raw, dict):
            raw = {int(k):v for k,v in raw.items()}
            if sorted(raw) != list(range(len(raw))):
                raise ValueError('类别编号必须从 0 开始连续排列')
            self.names = [str(raw[i]) for i in range(len(raw))]
        elif isinstance(raw, list):
            self.names = [str(v) for v in raw]
        elif classes.exists():
            self.names = classes.read_text(encoding='utf-8-sig').splitlines()
        else:
            self.names = []
        if self.names and (any(not n.strip() for n in self.names) or len(set(self.names)) != len(self.names)):
            raise ValueError('类别名称为空或重复，请检查类别配置')
        image_root = self.root/'images' if (self.root/'images').is_dir() else self.root
        label_root = self.root/'labels' if (self.root/'labels').is_dir() or image_root.name == 'images' else self.root
        files = sorted(p for p in image_root.rglob('*') if p.suffix.lower() in IMAGE_EXTS and not any(v.startswith('.') for v in p.relative_to(image_root).parts))
        self.samples = [Sample(p, label_root/p.relative_to(image_root).with_suffix('.txt'), p.relative_to(image_root).as_posix()) for p in files]
        if not self.samples:
            raise ValueError('未找到图片。请选择包含 images/labels 的数据集目录，或图片与标签同目录的文件夹。')
        destinations = [s.label for s in self.samples]
        if len(set(destinations)) != len(destinations):
            raise ValueError('存在同目录同名但不同扩展名图片，它们会共用标签；请先重命名处理')
        self.metadata_tokens = {p: fingerprint(p) for p in (self.yaml_path, classes)}

    def backup(self, path):
        if path.exists():
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                relative = Path('metadata')/path.name
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            target = self.root/'.annotation_backups'/stamp/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            return target

    def save_boxes(self, sample, boxes, token):
        if fingerprint(sample.label) != token:
            raise RuntimeError('此标签已被其他程序修改。请重新加载图片后再编辑，避免覆盖外部修改。')
        lines = []
        for box in boxes:
            box.validate(len(self.names))
            lines.append(f'{box.cls} {(box.x1+box.x2)/2:.8f} {(box.y1+box.y2)/2:.8f} {box.x2-box.x1:.8f} {box.y2-box.y1:.8f}')
        data = ('\n'.join(lines)+ ('\n' if lines else '')).encode('utf-8')
        if token is not None:
            self.backup(sample.label)
        atomic_write(sample.label, data)
        sample.status = '有标注' if boxes else '空标签'
        sample.detail = f'{len(boxes)} 个框'
        return data

    def save_names(self, names):
        names = [s.strip() for s in names]
        if len(names) < len(self.names) or any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError('类别名必须非空且唯一；不能删除已有类别编号')
        classes = self.root/'classes.txt'
        for p, token in self.metadata_tokens.items():
            if fingerprint(p) != token:
                raise RuntimeError('类别配置已被其他程序修改，请重新打开数据集')
        config = dict(self.config)
        config['names'] = dict(enumerate(names))
        config['nc'] = len(names)
        updates = {self.yaml_path: yaml.safe_dump(config,allow_unicode=True,sort_keys=False).encode('utf-8'), classes: ('\n'.join(names)+'\n').encode('utf-8')}
        previous = {p:fingerprint(p) for p in updates}
        completed = []
        try:
            for p,data in updates.items():
                self.backup(p)
                atomic_write(p,data)
                completed.append(p)
        except Exception:
            for p in reversed(completed):
                if previous[p] is None:
                    p.unlink(missing_ok=True)
                else:
                    atomic_write(p,previous[p])
            raise
        self.names = names
        self.config = config
        self.metadata_tokens = {p:fingerprint(p) for p in updates}

    def inspect(self, sample):
        try:
            if not sample.label.exists():
                sample.status, sample.detail = '缺失标签', '保存后可创建标签；不要把未标注图片自动当作背景'
            else:
                boxes = read_boxes(sample.label,len(self.names))
                sample.status, sample.detail = ('有标注' if boxes else '空标签'), f'{len(boxes)} 个框'
        except Exception as exc:
            sample.status, sample.detail = '格式异常', str(exc)
        return sample.status
