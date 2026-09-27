from src.evaluation.detection_metrics import calculate_iou 
 
def test_perfect_iou(): 
    box = [10, 10, 100, 100] 
    assert calculate_iou( 
        box, 
        box 
    ) == 1.0 
 
def test_no_overlap(): 
    box1 = [0, 0, 10, 10] 
    box2 = [20, 20, 30, 30] 
    assert calculate_iou( 
        box1, 
        box2 
    ) == 0.0