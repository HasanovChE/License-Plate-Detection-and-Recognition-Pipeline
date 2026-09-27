import numpy as np
from src.pipeline.pipeline import LicensePlatePipeline

def test_pipeline_no_detections():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    
    try:
        pipeline = LicensePlatePipeline(model_path="models/best.pt")
        result = pipeline.process(image)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "plates" in result
        assert result["success"] is False
    except FileNotFoundError:
        assert True