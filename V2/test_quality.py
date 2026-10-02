import itertools
import json
import random
from copy import deepcopy
from unittest.mock import patch
from pathlib import Path
import unittest
from quality import assignment,assess,evaluate_file,image_stamp,digest
from dataset import Box
from test_editor import Fixture,_APP
from app import MainWindow
from PyQt5.QtCore import Qt

def pred(cls=0,coords=(.1,.1,.3,.3),confidence=.9):return dict(cls=cls,xyxy=list(coords),confidence=confidence)

class ScoringTests(unittest.TestCase):
    def test_perfect_missing_extra_wrong_class(self):
        gt=[Box(0,.1,.1,.3,.3)]
        self.assertEqual(assess(gt,[pred()])['score'],100)
        self.assertEqual(assess(gt,[])['score'],0)
        self.assertAlmostEqual(assess(gt,[pred(),pred()])['score'],66.67)
        wrong=assess(gt,[pred(1)])
        self.assertEqual(wrong['score'],0);self.assertEqual(len(wrong['class_mismatches']),1)
    def test_position_and_empty(self):
        gt=[Box(0,.1,.1,.3,.3)]
        score=assess(gt,[pred(coords=(.2,.1,.4,.3))])
        self.assertAlmostEqual(score['score'],33.33);self.assertEqual(score['shifted_labels'],[0])
        self.assertEqual(assess([],[])['score'],100)
        self.assertEqual(assess([],[pred()])['score'],0)
    def test_confidence_is_not_quality(self):
        self.assertEqual(assess([Box(0,.1,.1,.3,.3)],[pred(confidence=.26)])['score'],100)
    def test_one_prediction_cannot_match_two_labels(self):
        result=assess([Box(0,.1,.1,.3,.3)]*2,[pred()])
        self.assertEqual(len(result['matches']),1);self.assertEqual(result['score'],66.67)
    def test_optimal_assignment_against_brute_force(self):
        rng=random.Random(42)
        for rows in range(1,5):
            for cols in range(1,5):
                weights=[[rng.random() for _ in range(cols)] for _ in range(rows)]
                n=max(rows,cols)
                padded=[[weights[i][j] if i<rows and j<cols else 0 for j in range(n)] for i in range(n)]
                best=max(sum(padded[i][j] for i,j in enumerate(order)) for order in itertools.permutations(range(n)))
                actual=sum(weights[i][j] for i,j in assignment(weights))
                self.assertAlmostEqual(actual,best)

class ReviewTests(Fixture):
    def setUp(self):
        super().setUp();self.window=MainWindow();self.window.error=lambda e:self.fail(str(e));self.window.show();self.window.open_dataset(self.root);_APP.processEvents()
        w=self.window;w.model_info={'names':{0:'motor_rotor',1:'long_motor'},'model_digest':'fake'};w.model_path.setText('/tmp/fake.pt');w.auto_mapping();w.context=w.make_context()
    def tearDown(self):
        w=self.window;w.saved=deepcopy(w.canvas.boxes);w.repair=False;w.close();super().tearDown()
    def review(self,index,predictions):
        s=self.window.dataset.samples[index]
        r=evaluate_file(s.label,predictions,self.ds.names);r.update(predictions=predictions,image_stamp=image_stamp(s.image));return r
    def test_missing_label_is_zero_even_without_predictions(self):
        self.assertEqual(self.review(1,[])['score'],0)
    def test_sort_filter_navigation_save_correct_target(self):
        w=self.window;w.reviews={0:self.review(0,[]),1:self.review(1,[])}
        w.reviews[0]['score']=95;w.reviews[1]['score']=10
        w.refresh_reviews();w.sort_reviews()
        self.assertEqual(w.images.item(0).data(Qt.UserRole),1)
        w.low_only.setChecked(True);self.assertTrue(w.sample_items[0].isHidden());self.assertFalse(w.sample_items[1].isHidden())
        w.jump_low();self.assertEqual(w.active_index,1)
        a_before=self.label.read_bytes()
        self.assertTrue(w.save());self.assertEqual(self.label.read_bytes(),a_before)
        self.assertEqual((self.root/'labels/train/b.txt').read_text(),'')
        self.assertEqual(w.reviews[1]['score'],100)
        w.low_only.setChecked(False);w.low_first.setChecked(False)
        self.assertEqual(w.images.item(0).data(Qt.UserRole),0)
        w.low_first.setChecked(True)
        self.assertEqual(w.images.item(0).data(Qt.UserRole),0) # 95 < refreshed 100
    def test_edit_save_recomputes_and_caches(self):
        w=self.window;w.reviews={0:self.review(0,[pred(0,(.4,.4,.6,.6)),pred(1,(.15,.15,.25,.25))])}
        w.refresh_reviews();self.assertEqual(w.reviews[0]['score'],100)
        w.select_box(0);w.delete_box();w.save()
        self.assertEqual(w.reviews[0]['score'],66.67)
        cached=json.loads((self.root/'.model_review/latest.json').read_text())
        self.assertEqual(cached['results']['train/a.jpg']['score'],66.67)
        w.reviews={}
        with patch('review_ui.QFileDialog.getOpenFileName',return_value=(str(self.root/'.model_review/latest.json'),'')):
            w.import_results()
        self.assertEqual(w.reviews[0]['score'],66.67)
    def test_external_label_change_recomputed_and_image_stale(self):
        w=self.window;w.reviews={0:self.review(0,[])}
        self.label.write_text('');w.ensure_fresh(0);self.assertEqual(w.reviews[0]['score'],100)
        with w.dataset.samples[0].image.open('ab') as f:f.write(b'changed')
        w.ensure_fresh(0);self.assertIsNone(w.reviews[0]['score']);self.assertTrue(w.reviews[0]['stale'])
    def test_parameters_clear_results_threshold_does_not(self):
        w=self.window;w.reviews={0:self.review(0,[])}
        w.quality_threshold.setValue(50);self.assertEqual(len(w.reviews),1)
        w.confidence.setValue(.4);self.assertEqual(w.reviews,{})
    def test_sorted_autosave_navigation_keeps_selection(self):
        w=self.window;w.reviews={0:self.review(0,[]),1:self.review(1,[])}
        w.refresh_reviews();w.sort_reviews();w.autosave.setChecked(True)
        w.select_box(0);w.delete_box()
        w.images.setCurrentItem(w.sample_items[1])
        self.assertEqual(w.active_index,1)
        self.assertEqual(w.images.currentItem().data(Qt.UserRole),1)
        self.assertEqual(w.sample.image.name,'b.jpg')
    def test_mapping_by_name_not_numeric_id(self):
        w=self.window;w.model_info['names']={0:'long_motor',1:'motor_rotor'};w.auto_mapping()
        self.assertEqual(w.mapping,{0:1,1:0});self.assertTrue(w.validate_mapping())
    def test_switch_dataset_clears_results(self):
        w=self.window;w.reviews={9000:dict(score=3)} # prior dataset can have more images
        w.open_dataset(self.root);self.assertFalse(w.reviews)

if __name__=='__main__':unittest.main(verbosity=2)
