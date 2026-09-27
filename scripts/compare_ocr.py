def compare_ocr( 
    test_images, 
    tesseract, 
    easyocr 
): 
 
    rows = [] 
 
    for image in test_images: 
 
        tess_text = tesseract.read(image) 
        easy_text = easyocr.read(image) 
 
        rows.append({ 
            "image": image, 
            "tesseract": tess_text, 
            "easyocr": easy_text 
        }) 
    return rows