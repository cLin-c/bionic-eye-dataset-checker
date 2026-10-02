"""Tests run on temporary datasets; original training data is never modified."""
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import yaml
from dataset import Dataset, Box, read_boxes, fingerprint

class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        (self.root/'images/train').mkdir(parents=True);(self.root/'labels/train').mkdir(parents=True)
        Image.new('RGB',(640,480),'#405060').save(self.root/'images/train/a.jpg')
        Image.new('RGB',(640,480),'#506070').save(self.root/'images/train/b.jpg')
        self.label=self.root/'labels/train/a.txt';self.label.write_text('0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n')
        self.config={'path':str(self.root),'train':'images/train','val':'images/train','names':{0:'motor_rotor',1:'long_motor'},'nc':2}
        (self.root/'data.yaml').write_text(yaml.safe_dump(self.config))
        self.ds=Dataset(self.root)
    def tearDown(self):self.temp.cleanup()

class IOTests(Fixture):
    def test_roundtrip_backup_and_conflict(self):
        sample=self.ds.samples[0];token=fingerprint(sample.label);boxes=read_boxes(sample.label,2)
        boxes[0]=Box(1,0.1,0.2,0.8,0.9)
        new=self.ds.save_boxes(sample,boxes,token)
        loaded=read_boxes(sample.label,2)
        for actual,expected in zip(loaded,boxes):
            self.assertEqual(actual.cls,expected.cls)
            for key in ('x1','y1','x2','y2'):
                self.assertAlmostEqual(getattr(actual,key),getattr(expected,key),places=7)
        backups=list((self.root/'.annotation_backups').rglob('a.txt'))
        self.assertEqual(len(backups),1);self.assertEqual(backups[0].read_bytes(),token)
        sample.label.write_text('external change')
        with self.assertRaises(RuntimeError):self.ds.save_boxes(sample,[],new)
        self.assertEqual(sample.label.read_text(),'external change')
    def test_malformed_not_dropped(self):
        for text in ['0 .5 .5 .2 .2\nbad line','2 .5 .5 .2 .2','-1 .5 .5 .2 .2','0 nan .5 .1 .1','0 .5 .5 -1 .2','0 .99 .5 .3 .2','0 .5 .5 .2 .2 .1']:
            self.label.write_text(text)
            with self.assertRaises(ValueError):read_boxes(self.label,2)
            self.assertEqual(self.ds.inspect(self.ds.samples[0]),'格式异常')
    def test_missing_empty_are_distinct(self):
        self.assertEqual(self.ds.inspect(self.ds.samples[1]),'缺失标签')
        self.ds.save_boxes(self.ds.samples[1],[],None)
        self.assertEqual(self.ds.inspect(self.ds.samples[1]),'空标签')
    def test_class_names_preserve_ids_and_paths(self):
        before=self.label.read_bytes()
        self.ds.save_names(['转子','长电机','其他'])
        cfg=yaml.safe_load((self.root/'data.yaml').read_text())
        self.assertEqual(cfg['train'],'images/train');self.assertEqual(cfg['names'],{0:'转子',1:'长电机',2:'其他'})
        self.assertEqual(self.label.read_bytes(),before)
        self.assertEqual(Dataset(self.root).names,['转子','长电机','其他'])
        with self.assertRaises(ValueError):self.ds.save_names(['重复','重复','其他'])
        with self.assertRaises(ValueError):self.ds.save_names(['转子'])
    def test_metadata_concurrent_write(self):
        (self.root/'classes.txt').write_text('external\n')
        with self.assertRaises(RuntimeError):self.ds.save_names(['a','b'])
    def test_boundary_rounding(self):
        self.label.write_text('0 0.5 0.833333 1 0.333334\n')
        self.assertEqual(len(read_boxes(self.label,2)),1)
    def test_images_directory_mapping(self):
        ds=Dataset(self.root/'images')
        self.assertEqual(ds.samples[0].label,self.label)
    def test_filename_collision(self):
        Image.new('RGB',(640,480)).save(self.root/'images/train/a.png')
        with self.assertRaises(ValueError):Dataset(self.root)

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt,QPoint,QPointF,QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from app import MainWindow,STYLE
_APP=QApplication.instance() or QApplication([])
_APP.setStyleSheet(STYLE)

def drag(widget,start,end):
    start=QPoint(round(start.x()),round(start.y()));end=QPoint(round(end.x()),round(end.y()))
    QTest.mousePress(widget,Qt.LeftButton,pos=start)
    event=QMouseEvent(QEvent.MouseMove,QPointF(end),Qt.NoButton,Qt.LeftButton,Qt.NoModifier)
    QApplication.sendEvent(widget,event)
    QTest.mouseRelease(widget,Qt.LeftButton,pos=end)
    QApplication.processEvents()

class UITests(Fixture):
    def setUp(self):
        super().setUp();self.window=MainWindow();self.window.show();self.window.open_dataset(self.root);QApplication.processEvents()
    def tearDown(self):
        self.window.saved=deepcopy(self.window.canvas.boxes);self.window.repair=False;self.window.close();super().tearDown()
    def test_draw_move_resize_delete_undo_save_reload(self):
        w=self.window;c=w.canvas
        self.assertEqual(len(c.boxes),2)
        w.actions['draw'].setChecked(True);w.toggle_draw(True)
        drag(c,c.screen(.65,.65),c.screen(.9,.9))
        self.assertEqual(len(c.boxes),3);self.assertTrue(w.dirty())
        w.toggle_draw(False)
        old=deepcopy(c.boxes[2]);drag(c,c.screen(.77,.77),c.screen(.70,.70))
        self.assertLess(c.boxes[2].x1,old.x1)
        old=deepcopy(c.boxes[2]);drag(c,c.handles(c.boxes[2])[4],c.screen(.95,.95))
        self.assertGreater(c.boxes[2].x2,old.x2)
        w.class_combo.setCurrentIndex(1);self.assertEqual(c.boxes[2].cls,1)
        before=deepcopy(c.boxes);w.delete_box();self.assertEqual(len(c.boxes),2)
        w.undo();self.assertEqual(c.boxes,before);w.redo();self.assertEqual(len(c.boxes),2)
        w.undo();self.assertTrue(w.save());self.assertFalse(w.dirty())
        w.load_sample(0);self.assertEqual(len(c.boxes),3);self.assertEqual(c.boxes[2].cls,1)
    def test_coordinates_navigation_and_filter(self):
        w=self.window;w.select_box(0)
        for spin,v in zip(w.spins,[10,20,100,80]):spin.setValue(v)
        w.apply_coordinates();b=w.canvas.boxes[0]
        self.assertAlmostEqual(b.x1,10/640);self.assertAlmostEqual(b.y2,100/480)
        w.autosave.setChecked(True);w.navigate(1)
        self.assertEqual(w.active_index,1);self.assertEqual(w.sample.status,'缺失标签')
        self.assertTrue(w.save());self.assertEqual(w.sample.status,'空标签')
        w.filter.setCurrentText('空标签')
        self.assertTrue(w.images.item(0).isHidden());self.assertFalse(w.images.item(1).isHidden())
    def test_invalid_label_blocks_save(self):
        self.label.write_text('broken annotation')
        self.window.load_sample(0)
        self.assertTrue(self.window.blocked);self.assertFalse(self.window.actions['save'].isEnabled())
        self.assertEqual(self.label.read_text(),'broken annotation')

if __name__=='__main__':unittest.main(verbosity=2)
