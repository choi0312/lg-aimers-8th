# LG Aimers 8th: EXAONE LLM Compression

LG Aimers 8기 **모델 경량화 온라인 해커톤**에서 사용한 EXAONE-4.0-1.2B 기반 LLM 경량화 파이프라인입니다.

본 저장소는 기존 Colab 노트북을 그대로 업로드하지 않고, calibration dataset 구성, 모델 로딩, W8A8 static PTQ, sanity check, 제출용 `submit.zip` 생성 과정을 모듈형 Python 프로젝트로 재구성한 버전입니다.

## Competition Result

| 항목 | 내용 |
|---|---|
| 대회명 | Aimers 8기 : 모델 경량화 온라인 해커톤 |
| 플랫폼 | DACON |
| 주제 | LLM 경량화, Large Language Model Compression |
| 기본 모델 | LGAI-EXAONE/EXAONE-4.0-1.2B |
| 제출 형식 | HuggingFace 표준 모델 디렉토리를 `submit.zip/model/` 구조로 제출 |
| 리더보드 기록 | 0.63147 |
| 대회 링크 | https://dacon.io/competitions/official/236673/overview/description |

## 1. 프로젝트 개요

본 프로젝트는 EXAONE-4.0-1.2B 모델을 대상으로, 성능 저하를 최소화하면서 추론 효율을 개선하기 위한 post-training quantization 기반 경량화 파이프라인입니다.

대회 평가 환경에서는 참가자의 코드가 직접 실행되는 것이 아니라, 제출된 HuggingFace 표준 모델 가중치와 설정 파일이 운영진의 고정 vLLM 추론 환경에서 실행됩니다. 따라서 본 프로젝트는 모델 가중치와 config 수준에서 호환 가능한 결과물을 생성하는 데 초점을 둡니다.

## 2. 접근 전략

| 구분 | 내용 |
|---|---|
| Base Model | LGAI-EXAONE/EXAONE-4.0-1.2B |
| Compression | W8A8 static post-training quantization |
| Quantization Tooling | llmcompressor + compressed-tensors |
| Calibration | MANTA, KMMLU-Redux, GSM8K, KoMT-Bench 혼합 prompt 512개 |
| Sequence Length | 최대 1024 tokens 기준 calibration |
| Ignore Policy | `lm_head`는 양자화에서 제외해 출력 분포 안정성 유지 |
| Submission | 최상위 경로에 `model/`만 포함된 submit.zip 생성 |
| Sanity Check | 오프라인 모델 로딩, chat template 적용, 간단한 생성 품질 점검 |

## 3. 핵심 설계 의도

### 3.1 Calibration Mix

Calibration prompt는 한국어 지식형 문항, 수학 추론, 일반 대화형 지시, 멀티턴 한국어 benchmark를 혼합했습니다. 이는 단일 데이터셋에 과적합된 activation range를 피하고, EXAONE 모델이 평가받을 수 있는 다양한 prompt 분포를 커버하기 위한 설계입니다.

### 3.2 Static W8A8 PTQ

모델의 Linear layer에 대해 weight와 activation 모두 INT8 양자화를 적용하는 W8A8 PTQ 구성을 사용했습니다. 단순 weight-only 압축보다 vLLM 환경에서 추론 효율 개선 가능성이 크며, calibration을 통해 activation clipping range를 안정화하는 것이 핵심입니다.

### 3.3 `lm_head` 제외

출력 vocabulary projection에 해당하는 `lm_head`는 모델 응답 품질과 형식 안정성에 직접적인 영향을 미칠 수 있으므로 양자화 대상에서 제외했습니다. 이를 통해 압축률과 생성 안정성 사이의 균형을 확보했습니다.

### 3.4 제출 구조 검증

대회 제출물은 `submit.zip` 내부 최상위에 `model/` 디렉토리만 존재해야 합니다. 본 프로젝트는 패키징 단계에서 zip 내부 구조를 자동 검증하여 설치 오류를 방지합니다.

## 4. 저장소 구조

<pre>
lg-aimers-8th/
├─ configs/
│  └─ default.yaml
├─ src/
│  └─ lg_aimers_8th/
│     ├─ config.py
│     ├─ calibration.py
│     ├─ model_utils.py
│     ├─ quantization.py
│     ├─ packaging.py
│     ├─ sanity.py
│     └─ pipeline.py
├─ scripts/
│  ├─ run_ptq.py
│  └─ package_submit.py
├─ docs/
├─ tests/
├─ README.md
└─ requirements.txt
</pre>

## 5. 실행 방법

### 5.1 HuggingFace Token 설정

<pre>
export HF_TOKEN="your_huggingface_token"
</pre>

### 5.2 전체 PTQ 파이프라인 실행

<pre>
python scripts/run_ptq.py --config configs/default.yaml
</pre>

### 5.3 제출 파일만 다시 생성

<pre>
python scripts/package_submit.py --model_dir /content/model --zip_path /content/submit.zip
</pre>

## 6. 출력 파일

<pre>
/content/model/
├─ config.json
├─ generation_config.json
├─ tokenizer.json
├─ tokenizer_config.json
├─ model.safetensors 또는 압축 모델 파일
└─ 기타 HuggingFace 표준 모델 파일

/content/submit.zip
└─ model/
</pre>

## 7. 주의사항

- 본 저장소에는 실제 모델 가중치와 submit.zip을 포함하지 않습니다.
- 모델 파일은 용량이 크므로 GitHub에는 업로드하지 않고, 실행 환경에서 생성한 뒤 대회 제출용 zip으로 관리합니다.
- 평가 서버는 인터넷 접속이 불가능하므로 제출 zip 내부의 `model/`만으로 tokenizer와 model이 모두 로드 가능해야 합니다.
- vLLM 라이브러리 자체 수정은 온라인 해커톤 조건에서 허용되지 않으므로, 모델 가중치와 config 수준의 결과물만 생성합니다.
