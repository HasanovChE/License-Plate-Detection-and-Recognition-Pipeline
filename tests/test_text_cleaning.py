from src.utils.text import clean_ocr_text 
 
def test_clean_text(): 
    result = clean_ocr_text( 
        "99-ab-123\n" 
    ) 
    assert result == "99AB123"