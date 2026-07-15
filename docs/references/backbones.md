---
tags: [roi-corner-detection, backbones, pretrained-weights]
status: reference
created: 2026-07-12
updated: 2026-07-15
---

# Backbone 가중치 카탈로그

이 문서는 `/mnt/d/backbones`에 있는 46개 가중치의 파일 정보, 직접 URL과 SHA-256을 관리하는
reference catalog다. Architecture, method registry와 실험 우선순위는
[model-assembly.md](../architecture/model-assembly.md)를 따른다. 다른 PC에서는 표의 직접 URL로
같은 파일을 받고 SHA-256을 비교한다.

## 1. 분류 기준과 사용 원칙

목표 영상은 stage 위 rounded OLED panel의 단일 방향 fringe 영상이다. 네 레이블은 실제 sharp
pixel corner가 아니라 네 직선 변을 원호 너머로 연장한 가상 교점이다. 내부 fringe는 panel
영역을 가르는 유용한 단서지만 raw line detector에는 다수의 false line을 만든다.

표의 분류 기준은 다음과 같다.

| 분류 | 의미 |
|---|---|
| 권장 | `reg`, `heatmap`, `seg`, `det` 또는 rule-based comparison에 바로 연결할 가치가 있다. |
| 조건부 | architecture, pretraining task, 모델 크기 또는 fringe 도메인 불일치 때문에 별도 ablation이나 adapter가 필요하다. |
| 비권장 또는 비관련 | 현재 ROI corner detection의 model 초기화나 postprocess에 직접 사용하지 않는다. |

가중치 파일은 다음처럼 검증한다. 다운로드 도구와 저장 경로는 환경에 맞게 바꿀 수 있다.

```bash
curl -L --fail --silent --show-error -o <filename> <direct-url>
sha256sum <filename>
```

URL 열에서 `검증된 직접 URL 없음`은 로컬 파일의 정확한 upstream release를 신뢰성 있게 특정하지
못했다는 뜻이다. 이 경우 유사한 이름의 checkpoint를 대체 다운로드하면 state dict, license 또는
pretraining이 달라질 수 있으므로 권장 방법에 사용하지 않는다.

## 2. 권장 가중치

다음 가중치는 현재 구현 우선순위의 `reg`, frozen DINOv2 backbone variant, `seg`
또는 geometry postprocess comparison에 연결하기 적합하다. Segmentation 계열은 COCO head를 binary panel mask head로
교체하고, mask는 four-side line fitting을 거쳐 가상 corner로 변환한다.

| 로컬 파일과 크기 | architecture와 사전학습 | 적용 방법 | 직접 URL | SHA-256 |
|---|---|---|---|---|
| `resnet18-f37072fd.pth`<br>46,830,571 B | ResNet-18, ImageNet-1K | `reg`, `heatmap`, `seg` 기준 backbone | [PyTorch](https://download.pytorch.org/models/resnet18-f37072fd.pth) | `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| `resnet34-b627a593.pth`<br>87,319,819 B | ResNet-34, ImageNet-1K | ResNet depth ablation | [PyTorch](https://download.pytorch.org/models/resnet34-b627a593.pth) | `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f` |
| `resnet50-0676ba61.pth`<br>102,530,333 B | ResNet-50, ImageNet-1K | accuracy-oriented CNN backbone | [PyTorch](https://download.pytorch.org/models/resnet50-0676ba61.pth) | `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` |
| `mobilenet_v2-7ebf99e0.pth`<br>14,258,573 B | MobileNetV2, ImageNet-1K | CPU latency baseline | [PyTorch](https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth) | `7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f` |
| `mobilenet_v3_small-047dcff4.pth`<br>10,306,551 B | MobileNetV3-Small, ImageNet-1K | 최소 크기 CPU baseline | [PyTorch](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) | `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f` |
| `mobilenet_v3_large-8738ca79.pth`<br>22,139,423 B | MobileNetV3-Large, ImageNet-1K | 경량 accuracy baseline | [PyTorch](https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth) | `8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04` |
| `efficientnet_b0_rwightman-7f5810bc.pth`<br>21,444,401 B | EfficientNet-B0, ImageNet-1K | CNN architecture ablation | [PyTorch](https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth) | `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934` |
| `vit_b_16-c867db91.pth`<br>346,328,529 B | ViT-B/16, ImageNet-1K | `reg` ViT backbone variant | [PyTorch](https://download.pytorch.org/models/vit_b_16-c867db91.pth) | `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9` |
| `swin_t-704ceda3.pth`<br>113,445,839 B | Swin-T, ImageNet-1K | `reg` multi-scale backbone variant | [PyTorch](https://download.pytorch.org/models/swin_t-704ceda3.pth) | `704ceda373461b0a224fcdddd75cd2a5e9f8064512ed47adbddef7f343fd147b` |
| `dinov2_vits14_pretrain.pth`<br>88,283,115 B | DINOv2 ViT-S/14, self-supervised | frozen `reg` adapter의 우선 후보 | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| `dinov2_vitb14_pretrain.pth`<br>346,378,731 B | DINOv2 ViT-B/14, self-supervised | frozen `reg` accuracy comparison | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | `0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73` |
| `fcn_resnet50_coco-1167a1af.pth`<br>141,567,418 B | FCN ResNet-50, COCO segmentation | `seg`, `usage=whole_model` variant | [PyTorch](https://download.pytorch.org/models/fcn_resnet50_coco-1167a1af.pth) | `1167a1affa42e1e62858f8d3fac12d109e0108327ffc91c5855a324b11683c36` |
| `deeplabv3_resnet50_coco-cd0a2569.pth`<br>168,312,152 B | DeepLabV3 ResNet-50, COCO segmentation | high-capacity `seg`, `usage=whole_model` variant | [PyTorch](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth) | `cd0a25694c4a0f7106b38f4938bf90a874f2f241cc410b8f63c7024399538f06` |
| `deeplabv3_mobilenet_v3_large-fc3c493d.pth`<br>44,356,159 B | DeepLabV3 MobileNetV3-Large, COCO segmentation | CPU-oriented `seg`, `usage=whole_model` variant | [PyTorch](https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth) | `fc3c493d68e89cc31ef488c803d5d7dd2f3190fb570598faa49fef69be8e5e70` |
| `lraspp_mobilenet_v3_large-d234d4ea.pth`<br>13,097,061 B | LR-ASPP MobileNetV3-Large, COCO segmentation | 최소 비용 `seg`, `usage=whole_model` 또는 geometry postprocess comparison | [PyTorch](https://download.pytorch.org/models/lraspp_mobilenet_v3_large-d234d4ea.pth) | `d234d4eae9d55d5f76de18b77cf0dc62c66fe5c5482758209d00f950c92bb280` |

## 3. 조건부 가중치

다음 가중치는 비교 실험, adapter 또는 특수 목적에는 쓸 수 있지만 현재 PMD fringe ROI의 첫
구현 대상으로는 우선하지 않는다. detection 모델은 일반 COCO object detection 가중치이므로
네 가상 corner를 직접 출력하지 않으며 별도 label 변환과 head fine-tuning이 필요하다.

### 3.1. Transformer와 CNN 비교 후보

다음 파일은 큰 모델, 중복 architecture 또는 pretraining 차이를 분석할 때 사용한다.

| 로컬 파일과 크기 | architecture와 사전학습 | 적용 방법과 제약 | 직접 URL | SHA-256 |
|---|---|---|---|---|
| `alexnet-owt-7be5be79.pth`<br>244,408,911 B | AlexNet, ImageNet-1K | legacy direct baseline, modern backbone 대비 효율이 낮음 | [PyTorch](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) | `7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02` |
| `efficientnet_b0_ra-3dd342df.pth`<br>21,376,743 B | EfficientNet-B0 RA variant | pretraining ablation, exact upstream release 미확정 | 검증된 직접 URL 없음 | `3dd342dfa1fee25ae65e7bbdf8998cad6e45d6e77e69d580f0bd14d3eeb0b3f3` |
| `efficientnet_b5_lukemelas-1a07897c.pth`<br>122,540,693 B | EfficientNet-B5 | 대형 CNN ablation, exact upstream release 미확정 | 검증된 직접 URL 없음 | `1a07897c0d357db7981640f6be44a63420f11deb932344a69768b62ebe272946` |
| `squeezenet1_1-b8a52dc0.pth`<br>4,958,839 B | SqueezeNet 1.1, ImageNet-1K | 극소형 reference, 정확도 위험이 큼 | [PyTorch](https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth) | `b8a52dc049b60e4b6ab68ad0df457362afab8b6304b2febdc1650a5dab4d7e7b` |
| `vgg16-397923af.pth`<br>553,433,881 B | VGG-16, ImageNet-1K | legacy feature baseline, F6에 불리 | [PyTorch](https://download.pytorch.org/models/vgg16-397923af.pth) | `397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0` |
| `vgg16_bn-6c64b313.pth`<br>553,507,836 B | VGG-16-BN, ImageNet-1K | legacy feature baseline, F6에 불리 | [PyTorch](https://download.pytorch.org/models/vgg16_bn-6c64b313.pth) | `6c64b3138f2f4fcb3bcc4cafde11619c4f440eb1631787e93a682fd88305888a` |
| `vgg19-dcbb9e9d.pth`<br>574,673,361 B | VGG-19, ImageNet-1K | legacy feature baseline, F6에 불리 | [PyTorch](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) | `dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44` |
| `vgg19_bn-c79401a0.pth`<br>574,769,405 B | VGG-19-BN, ImageNet-1K | legacy feature baseline, F6에 불리 | [PyTorch](https://download.pytorch.org/models/vgg19_bn-c79401a0.pth) | `c79401a0cf3cb42714e4182f5868c7a6f4f4534f5df9e956e2bb2098de41cbb6` |
| `wide_resnet50_2-95faca4d.pth`<br>138,223,492 B | Wide-ResNet-50-2, ImageNet-1K | wide CNN backbone ablation | [PyTorch](https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth) | `95faca4d11227dddf8633dbb5ff6c8a9003c1aa5b8945c73834b8007b10950b8` |
| `wide_resnet101_2-32ee1156.pth`<br>254,695,146 B | Wide-ResNet-101-2, ImageNet-1K | 대형 CNN upper-bound, CPU 배포에 부적합 | [PyTorch](https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth) | `32ee11565f95f2f07223b243ada444a6457c4b90a88e6e293922813faf6dfd06` |
| `wide_resnet50_2.tv_in1k/model.safetensors`<br>275,835,296 B | Wide-ResNet-50-2, ImageNet-1K<br>timm ID `wide_resnet50_2.tv_in1k` | PyTorch weight와 별도 timm serialization 비교 | [Hugging Face](https://huggingface.co/timm/wide_resnet50_2.tv_in1k/resolve/main/model.safetensors) | `df6fb6c4824769769de18e14088475fd6ee94236849aa4e5d8022ba9d9a9a16c` |
| `deit_base_distilled_patch16_224.fb_in1k/model.safetensors`<br>349,367,122 B | DeiT-Base distilled, ImageNet-1K, 224<br>timm ID `deit_base_distilled_patch16_224.fb_in1k` | ViT direct alternative, distillation head 처리 필요 | [Hugging Face](https://huggingface.co/timm/deit_base_distilled_patch16_224.fb_in1k/resolve/main/model.safetensors) | `ccc9d1bbeede1fc8609a7a7482773a35057e9f38035b9b804bced9126c5a70dc` |
| `deit_base_distilled_patch16_384.fb_in1k/model.safetensors`<br>350,534,482 B | DeiT-Base distilled, ImageNet-1K, 384<br>timm ID `deit_base_distilled_patch16_384.fb_in1k` | 해상도 ablation, 224 공정 비교와 분리 | [Hugging Face](https://huggingface.co/timm/deit_base_distilled_patch16_384.fb_in1k/resolve/main/model.safetensors) | `f739dfae2bf3fdd4ef415fdb015966c515b9adca7de703a08365fb349fea82ca` |
| `cait_s24_224.fb_dist_in1k/model.safetensors`<br>187,709,078 B | CaiT-S24, ImageNet-1K, 224<br>timm ID `cait_s24_224.fb_dist_in1k` | Transformer architecture ablation | [Hugging Face](https://huggingface.co/timm/cait_s24_224.fb_dist_in1k/resolve/main/model.safetensors) | `ec4c0b0e1851c9b1850709caf5cbbcb0ecf121efb4317eedd99ceb0e0e0f4ddb` |
| `cait_m48_448.fb_dist_in1k/model.safetensors`<br>1,425,928,240 B | CaiT-M48, ImageNet-1K, 448<br>timm ID `cait_m48_448.fb_dist_in1k` | 대형 Transformer upper-bound, F6에 부적합 | [Hugging Face](https://huggingface.co/timm/cait_m48_448.fb_dist_in1k/resolve/main/model.safetensors) | `460cbe2667f0208bd39ea38f981e5d83b41bfec33ed8cefab3460ae65f4788ed` |

### 3.2. DINOv2 register-token과 대형 변형

다음 파일은 frozen DINOv2 backbone ablation용이다. Register-token 유무는 같은 encoder 크기의 실험 축으로
취급하고, ViT-L은 CPU latency와 model size 제약을 만족할 가능성이 낮다.

| 로컬 파일과 크기 | architecture와 사전학습 | 적용 방법과 제약 | 직접 URL | SHA-256 |
|---|---|---|---|---|
| `dinov2_vits14_reg4_pretrain.pth`<br>88,291,785 B | DINOv2 ViT-S/14 reg4, self-supervised | register-token ablation | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth) | `f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb` |
| `dinov2_vitb14_reg4_pretrain.pth`<br>346,393,545 B | DINOv2 ViT-B/14 reg4, self-supervised | register-token ablation | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) | `73182a088cf94833c94b1666d1c99e02fe87e2007bff57b564fb6206e25dba71` |
| `dinov2_vitl14_pretrain.pth`<br>1,217,586,395 B | DINOv2 ViT-L/14, self-supervised | accuracy upper-bound, CPU 배포에 부적합 | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |
| `dinov2_vitl14_reg4_pretrain.pth`<br>1,217,607,321 B | DINOv2 ViT-L/14 reg4, self-supervised | register-token upper-bound | [Meta](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth) | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` |

### 3.3. Detection과 line 후보

다음 파일은 공통 corner contract로 변환하는 adapter가 있어야만 쓸 수 있다. 특히 M-LSD는 내부
fringe line을 우선 검출하므로 mask boundary band 또는 fringe suppression 없이 사용하지 않는다.

| 로컬 파일과 크기 | architecture와 사전학습 | 적용 방법과 제약 | 직접 URL | SHA-256 |
|---|---|---|---|---|
| `fasterrcnn_resnet50_fpn_coco-258fb6c6.pth`<br>167,502,836 B | Faster R-CNN ResNet-50-FPN, COCO detection | `det` corner box adapter, 4-class fine-tuning 필요 | [PyTorch](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth) | `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384` |
| `retinanet_resnet50_fpn_coco-eeacb38b.pth`<br>136,595,076 B | RetinaNet ResNet-50-FPN, COCO detection | `det` corner box adapter, 4-class fine-tuning 필요 | [PyTorch](https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth) | `eeacb38b7cec8cf93c57867e05eaab621047f19b0d2ec5accaa405f690da15b7` |
| `ssd300_vgg16_coco-b556d3b4.pth`<br>142,594,222 B | SSD300 VGG16, COCO detection | legacy corner box baseline, CPU 비용 큼 | [PyTorch](https://download.pytorch.org/models/ssd300_vgg16_coco-b556d3b4.pth) | `b556d3b43ab6c3f63d81bfb8835fe8756ac22da664357da100dccf96b6a6b42d` |
| `yolov8n.pt`<br>6,549,796 B | Ultralytics YOLOv8-Nano, COCO detection | `det` YOLO corner box fine-tuning | [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `detr-r50-e632da11.pth`<br>166,618,694 B | DETR ResNet-50, COCO detection | `det` DETR box 초기화와 조건부 point head 학습 | [Meta](https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth) | `e632da11ec76ae67bac2f8579fbed3724e08dead7d200ca13e019b197784eadc` |
| `mlsd_large_512_fp32.pth`<br>6,341,481 B | M-LSD large, line segment detection | `line` ablation만 가능, exact upstream release 미확정 | 검증된 직접 URL 없음 | `5696f168eb2c30d4374bbfd45436f7415bb4d88da29bea97eea0101520fba082` |

## 4. 비권장 또는 비관련 가중치

다음 파일은 현재 method catalog의 ROI corner model 초기화에 직접 대응하지 않는다. 보관은
유지하되, 다른 PC에서 프로젝트를 재현하기 위해 필수로 다운로드할 필요는 없다.

| 로컬 파일과 크기 | 원래 용도 | 제외 이유 | 직접 URL | SHA-256 |
|---|---|---|---|---|
| `lpips_alex.pth`<br>6,009 B | LPIPS Alex perceptual metric | image similarity metric용이며 ROI corner predictor가 아님 | [LPIPS](https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/alex.pth) | `df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0` |
| `lpips_squeeze.pth`<br>10,811 B | LPIPS Squeeze perceptual metric | image similarity metric용이며 ROI corner predictor가 아님 | [LPIPS](https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/squeeze.pth) | `4a5350f23600cb79923ce65bb07cbf57dca461329894153e05a1346bd531cf76` |
| `lpips_vgg.pth`<br>7,289 B | LPIPS VGG perceptual metric | image similarity metric용이며 ROI corner predictor가 아님 | [LPIPS](https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/weights/v0.1/vgg.pth) | `a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868` |
| `efficientad_pretrained_weights/pretrained_teacher_small.pth`<br>10,779,695 B | EfficientAD anomaly detection teacher | anomaly detection용 teacher이며 corner target이 없음 | 검증된 직접 URL 없음 | `a16ded54719674435576aee641152616a640dfc6dc2b83115dab6e226610ae7d` |
| `efficientad_pretrained_weights/pretrained_teacher_medium.pth`<br>32,110,817 B | EfficientAD anomaly detection teacher | anomaly detection용 teacher이며 corner target이 없음 | 검증된 직접 URL 없음 | `f7356663c8e00ada12ae01fb8c8aad0a1de2f800f8eadf252a46d29bbdfdf718` |
| `vq_model_pretrained_128_4096.pckl`<br>16,135,639 B | VQ generative or quantization model | ROI corner detection architecture와 대응하지 않음 | 검증된 직접 URL 없음 | `5efd46622ae8f44559b04312dab7a23fed669c394a816fb7928c3078e5877850` |

## 5. 파일 무결성 검증 결과

현재 `/mnt/d/backbones`에서 확장자 `.pth`, `.pt`, `.pckl`, `.safetensors`인 파일은 46개다.
각 표의 byte size와 SHA-256은 2026-07-12에 로컬 파일 전체를 다시 계산한 결과다. torchvision과
DETR 파일명에 포함된 짧은 hash는 배포 식별자이며, 표의 SHA-256은 전체 파일 검증값이다.

다른 PC에서 timm safe tensor artifact를 복원할 때는 표의 `model.safetensors`뿐 아니라 같은
Hugging Face model ID의 `config.json`도 함께 보관하는 것을 권장한다. 현재 로컬 디렉터리 이름이
model ID이며, canonical input size와 normalization은 해당 `config.json`에 기록되어 있다.
