import numpy as np 
from src.utils.image import crop_with_padding 
 
def test_crop(): 
    image = np.zeros( 
        (100, 100, 3), 
        dtype=np.uint8 
    ) 
    crop = crop_with_padding( 
        image, 
        [20, 20, 40, 40] 
    ) 
    assert crop is not None 
    assert crop.size > 0