from paddleocr import PaddleOCR

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _ocr


def ocr_pdf(pdf_path):
    results = _get_ocr().predict(str(pdf_path))
    pages = [" ".join(res["rec_texts"]) for res in results]
    return " ".join(pages)

# print(ocr_pdf(r"C:\Users\izdib\Projects\document_classifier\internal_docs\231101_Соглашение о конф_работник_Оразбай Әмірхан (1).pdf"))