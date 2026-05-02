# Modeling Strategy

## 1. Problem Formulation

본 해커톤은 LGAI-EXAONE/EXAONE-4.0-1.2B를 대상으로 모델 크기, 추론 속도, 성능 유지 사이의 균형을 최적화하는 문제이다. 평가 환경은 고정된 vLLM 기반 추론 서버이므로, 모델 구조 변경보다는 HuggingFace 표준 형식을 유지하는 경량화가 중요하다.

## 2. Quantization Strategy

W8A8 static PTQ를 사용한다. weight와 activation 모두 INT8로 압축하되, calibration prompt를 통해 activation observer가 안정적인 range를 추정하도록 한다.

## 3. Calibration Strategy

다음 네 가지 데이터 소스를 혼합한다.

- MANTA-1M: 일반 지시 및 대화형 prompt
- KMMLU-Redux: 한국어 지식형 객관식 문항
- GSM8K: 수학 추론 prompt
- KoMT-Bench: 한국어 멀티턴 benchmark

최종 calibration set은 512 samples, max sequence length 1024 기준으로 구성한다.

## 4. Stability Strategy

`lm_head`는 양자화에서 제외한다. 출력층의 dtype을 유지해 vocabulary projection의 수치 안정성을 보존하고, 모델이 반복 토큰이나 형식 붕괴를 일으키는 위험을 줄인다.

## 5. Submission Strategy

대회 제출 형식은 `submit.zip` 최상위에 `model/`만 포함되어야 한다. 패키징 단계에서 top-level folder를 자동 검사해 구조 오류를 사전에 차단한다.
