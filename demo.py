import torch
import torchvision
from torchvision import transforms as T
import cv2
import numpy as np
from PIL import Image


def load_image(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size)
    return img

def preprocess(img: Image.Image):
    # torchvision detection 모델들이 사용하는 transform
    transform = T.Compose([
        T.ToTensor(),
        # 필요하면 normalization 등 추가
    ])
    return transform(img)

def postprocess(outputs, score_thresh=0.5):
    """outputs: model(img_tensor.unsqueeze(0))의 결과 dict 리스트"""
    # outputs는 리스트 하나의 dict: {'boxes', 'labels', 'scores'}
    out = outputs[0]
    boxes = out['boxes'].detach().cpu().numpy()
    labels = out['labels'].detach().cpu().numpy()
    scores = out['scores'].detach().cpu().numpy()

    keep = scores >= score_thresh
    return boxes[keep], labels[keep], scores[keep]

def draw_boxes(img: np.ndarray, boxes, labels, scores, class_names=None):
    # img: numpy HWC uint8
    for box, lbl, sc in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        text = f"{class_names[lbl] if class_names else lbl}:{sc:.2f}"
        cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return img

def demo_fcos_image(image_path, device='cuda'):
    # 1. 모델 불러오기 (사전 학습된 FCOS)
    model = torchvision.models.detection.fcos_resnet50_fpn(weights="COCO_V1")
    model.to(device)
    model.eval()

    # 2. 이미지 로드 & 전처리
    img = load_image(image_path)
    img_tensor = preprocess(img).to(device)
    # 배치 차원 추가
    inputs = [img_tensor]

    # 3. 모델 추론
    with torch.no_grad():
        outputs = model(inputs)

    # 4. 결과 후처리
    boxes, labels, scores = postprocess(outputs, score_thresh=0.5)

    # 5. 시각화
    img_np = np.array(img)[:, :, ::-1].copy()  # PIL RGB -> OpenCV BGR
    img_with_boxes = draw_boxes(img_np, boxes, labels, scores)

    # 6. 결과 표시 혹은 저장
    cv2.imshow("FCOS Detection", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 혹은 저장
    # cv2.imwrite("output.jpg", img_with_boxes)

if __name__ == "__main__":
    demo_fcos_image("/home/mskim/D3T/cvpods/data/FLIR_ICIP2020_aligned/JPEGImages/FLIR_00002_PreviewData.jpeg")