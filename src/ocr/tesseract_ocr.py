import pytesseract


class TesseractOCR:
    def __init__(self, tesseract_path=None, psm_modes=(7, 6, 11)):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.psm_modes = psm_modes

    def _config(self, psm):
        return (
            "--oem 3 "
            f"--psm {psm} "
            "-c tessedit_char_whitelist="
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

    def read(self, image):
        """Try psm modes in order, return the first non-empty result."""
        for psm in self.psm_modes:
            text = pytesseract.image_to_string(
                image, config=self._config(psm)
            ).strip()
            if text:
                return text
        return ""
