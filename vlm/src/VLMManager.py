from typing import List
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import io
from PIL import Image
import torch

class VLMManager:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-base-patch16").to(self.device)
        self.processor = AutoProcessor.from_pretrained("google/owlv2-base-patch16")
        self.model.eval()
    def identify(self, image: bytes, caption: str) -> List[int]:
        im = Image.open(io.BytesIO(image))
        inputs = self.processor(text=[caption], images=im, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([im.size[::-1]]).to(self.device)
            results = self.processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]
        
        
        ratio = 1520/870
        boxes = results["boxes"].tolist()
        if(len(boxes) == 0):
            return [0,0,1520,870]
        scores = results["scores"].tolist()
        m = max(scores)
        xmin, ymin, xmax, ymax = boxes[scores.index(m)]
        ymin*=ratio
        ymax*=ratio
        return [int(xmin),int(ymin),int(xmax-xmin),int(ymax-ymin)]
