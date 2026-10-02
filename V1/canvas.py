"""Image-space annotation canvas. Coordinates are normalized, independent of zoom."""
from copy import deepcopy
from PyQt5.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QWidget
from dataset import Box

COLORS = ['#ff6b6b','#3dd6d0','#ffcf5c','#a89cff','#73df89','#f48bc1','#5aa9ff']

class Canvas(QWidget):
    selected = pyqtSignal(int)
    edited = pyqtSignal()
    coordinates = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.setMinimumSize(420,320)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.pixmap = QPixmap()
        self.boxes = []
        self.names = []
        self.index = -1
        self.scale = 1.
        self.offset = QPointF()
        self.draw_mode = False
        self.class_id = 0
        self.drag = None
        self.editable = True
        self.before = []
        self.start = QPointF()
        self.show_names = True

    def load(self,pixmap,boxes,names):
        self.pixmap,self.boxes,self.names=pixmap,boxes,names
        self.index=-1;self.drag=None
        self.fit()

    def fit(self):
        if self.pixmap.isNull():
            self.update();return
        self.scale = min((self.width()-30)/self.pixmap.width(),(self.height()-30)/self.pixmap.height())
        self.offset=QPointF((self.width()-self.pixmap.width()*self.scale)/2,(self.height()-self.pixmap.height()*self.scale)/2)
        self.update()

    def screen(self,x,y):
        return self.offset+QPointF(x*self.pixmap.width()*self.scale,y*self.pixmap.height()*self.scale)

    def point(self,p,clamp=True):
        x=(p.x()-self.offset.x())/(self.pixmap.width()*self.scale)
        y=(p.y()-self.offset.y())/(self.pixmap.height()*self.scale)
        return QPointF(min(1.,max(0.,x)),min(1.,max(0.,y))) if clamp else QPointF(x,y)

    def box_rect(self,b):
        return QRectF(self.screen(b.x1,b.y1),self.screen(b.x2,b.y2))

    def handles(self,b):
        r=self.box_rect(b)
        return [r.topLeft(),QPointF(r.center().x(),r.top()),r.topRight(),QPointF(r.right(),r.center().y()),r.bottomRight(),QPointF(r.center().x(),r.bottom()),r.bottomLeft(),QPointF(r.left(),r.center().y())]

    def paintEvent(self,event):
        p=QPainter(self);p.fillRect(self.rect_widget(),QColor('#101820'));p.setRenderHint(QPainter.Antialiasing)
        if self.pixmap.isNull():
            p.setPen(QColor('#a6b4c1'));p.drawText(self.rect_widget(),Qt.AlignCenter,'打开 YOLO 数据集开始检查');return
        dest=QRectF(self.offset,QPointF(self.offset.x()+self.pixmap.width()*self.scale,self.offset.y()+self.pixmap.height()*self.scale))
        p.drawPixmap(dest,self.pixmap,QRectF(self.pixmap.rect()))
        for i,b in enumerate(self.boxes):
            color=QColor(COLORS[b.cls%len(COLORS)])
            p.setPen(QPen(color,3 if i==self.index else 2));p.setBrush(Qt.NoBrush);r=self.box_rect(b);p.drawRect(r)
            if self.show_names:
                name=self.names[b.cls] if 0<=b.cls<len(self.names) else '未知'
                text=f'{i+1} · {b.cls}: {name}'
                metrics=p.fontMetrics();width=metrics.horizontalAdvance(text)+10;height=metrics.height()+6
                label=QRectF(r.left(),max(dest.top(),r.top()-height),width,height)
                p.fillRect(label,QColor(12,20,28,210));p.drawText(label.adjusted(5,0,-2,0),Qt.AlignVCenter,text)
            if i==self.index and self.editable:
                p.setBrush(color)
                for h in self.handles(b):p.drawRect(QRectF(h.x()-4,h.y()-4,8,8))
        if self.draw_mode:
            p.setPen(QColor('#ffcf5c'));p.drawText(14,24,'绘框模式：按住左键拖动，Esc 返回选择')

    def rect_widget(self):
        return super().rect()

    def mousePressEvent(self,e):
        self.setFocus()
        if self.pixmap.isNull():return
        if e.button()==Qt.MiddleButton or e.button()==Qt.RightButton:
            self.drag=('pan',e.pos(),QPointF(self.offset));return
        if e.button()!=Qt.LeftButton or not self.editable:return
        raw=self.point(e.pos(),False)
        self.start=self.point(e.pos());self.before=deepcopy(self.boxes)
        if self.draw_mode:
            if not (0<=raw.x()<=1 and 0<=raw.y()<=1):return
            self.boxes.append(Box(self.class_id,self.start.x(),self.start.y(),self.start.x(),self.start.y()))
            self.index=len(self.boxes)-1;self.drag=('draw',);self.selected.emit(self.index);self.update();return
        if 0<=self.index<len(self.boxes):
            for j,h in enumerate(self.handles(self.boxes[self.index])):
                if abs(e.x()-h.x())<=8 and abs(e.y()-h.y())<=8:
                    self.drag=('resize',j,deepcopy(self.boxes[self.index]));return
        hits=[i for i,b in enumerate(self.boxes) if self.box_rect(b).contains(e.pos())]
        self.index=min(hits,key=lambda i:(self.boxes[i].x2-self.boxes[i].x1)*(self.boxes[i].y2-self.boxes[i].y1)) if hits else -1
        self.selected.emit(self.index)
        if self.index>=0:self.drag=('move',deepcopy(self.boxes[self.index]))
        self.update()

    def mouseMoveEvent(self,e):
        if self.pixmap.isNull():return
        at=self.point(e.pos())
        self.coordinates.emit(f'x={at.x()*self.pixmap.width():.1f}  y={at.y()*self.pixmap.height():.1f}  缩放 {self.scale*100:.0f}%')
        if not self.drag:return
        mode=self.drag[0]
        if mode=='pan':
            self.offset=self.drag[2]+QPointF(e.pos()-self.drag[1]);self.update();return
        if not 0<=self.index<len(self.boxes):return
        b=self.boxes[self.index]
        if mode=='draw':
            b.x1,b.x2=sorted((self.start.x(),at.x()));b.y1,b.y2=sorted((self.start.y(),at.y()))
        elif mode=='move':
            orig=self.drag[1];dx=max(-orig.x1,min(1-orig.x2,at.x()-self.start.x()));dy=max(-orig.y1,min(1-orig.y2,at.y()-self.start.y()))
            b.x1,b.x2=orig.x1+dx,orig.x2+dx;b.y1,b.y2=orig.y1+dy,orig.y2+dy
        elif mode=='resize':
            j=self.drag[1];minx=1/self.pixmap.width();miny=1/self.pixmap.height()
            if j in (0,6,7):b.x1=min(at.x(),b.x2-minx)
            if j in (2,3,4):b.x2=max(at.x(),b.x1+minx)
            if j in (0,1,2):b.y1=min(at.y(),b.y2-miny)
            if j in (4,5,6):b.y2=max(at.y(),b.y1+miny)
        self.update()

    def mouseReleaseEvent(self,e):
        if not self.drag:return
        mode=self.drag[0];self.drag=None
        if mode=='pan':return
        if mode=='draw':
            b=self.boxes[self.index]
            if (b.x2-b.x1)*self.pixmap.width()<2 or (b.y2-b.y1)*self.pixmap.height()<2:
                self.boxes.pop();self.index=-1
        if self.before!=self.boxes:self.edited.emit()
        self.selected.emit(self.index);self.update()

    def wheelEvent(self,e):
        if self.pixmap.isNull():return
        pos=QPointF(e.pos());at=(pos-self.offset)/self.scale
        self.scale=max(.03,min(40.,self.scale*(1.18 if e.angleDelta().y()>0 else 1/1.18)))
        self.offset=pos-at*self.scale;self.update()

    def resizeEvent(self,e):
        self.fit()
