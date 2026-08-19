import paddle.inference
orig = paddle.inference.create_predictor

def patch(cfg): 
    cfg.delete_pass("self_attention_fuse_pass")
    return orig(cfg)

paddle.inference.create_predictor = patch

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False, enable_mkldnn=False, ir_optim=False)
print("OCR Initialized!")
