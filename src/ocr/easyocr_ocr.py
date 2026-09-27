import easyocr 
 
class EasyOCREngine: 
 
    def __init__(self): 
 
        self.reader = easyocr.Reader( 
            ["en"], 
            gpu=False 
        ) 
 
    def read(self, image): 
 
        results = self.reader.readtext( 
            image 
        ) 
 
        if not results: 
            return "" 
 
        results.sort( 
            key=lambda x: x[2], 
            reverse=True 
        ) 
 
        return results[0][1]