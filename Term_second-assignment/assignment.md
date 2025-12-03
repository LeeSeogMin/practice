# AI 기반 정책분석방법론 기말과제 2

## 과제명: 생성형 AI를 활용한 정책 예측 및 합성 데이터 분석

---

## 1. 과제 개요

### 1.1 배경

21세기 정책 환경은 기후 변화, 글로벌 팬데믹, 경제 위기 등 전례 없는 불확실성에 직면해 있다. 과거 데이터에 기반한 전통적 예측 방법론은 "무엇이 일어날 것인가?"에 답하지만, 생성형 AI는 "무엇이 일어날 수 있는가?"라는 가능성의 공간을 탐색하는 새로운 도구를 제공한다.

생성형 AI는 발생 가능한 수많은 미래 시나리오를 생성하고 그 분포를 분석함으로써 불확실성을 정량화하고 강건한 정책 설계를 가능하게 한다. 본 과제에서는 Transformer를 활용한 다변량 시계열 예측과 CTGAN을 활용한 합성 데이터 생성 및 품질 평가를 실습한다.

### 1.2 학습 목표

1. Transformer의 Self-Attention 메커니즘을 이해하고 다변량 시계열 예측에 적용할 수 있다
2. 정책 효과의 시차 구조(lag structure)를 분석할 수 있다
3. CTGAN을 활용하여 개인정보를 보호하면서 합성 데이터를 생성할 수 있다
4. 합성 데이터의 품질을 다양한 지표로 평가할 수 있다

---

## 2. 제공 데이터

data 폴더에 다음 2개의 CSV 파일이 제공된다:

### 2.1 policy_timeseries.csv (정책 시계열 데이터)

50개월간의 정책 투입 및 결과 데이터

| 변수명 | 설명 | 예시 |
|--------|------|------|
| month | 월 (1~50) | 1, 2, 3, ... |
| rd_budget | R&D 예산 (조원) | 2.5 |
| corporate_support | 기업 지원금 (조원) | 1.8 |
| tax_incentive | 세제 혜택 규모 (조원) | 0.9 |
| patent_count | 특허 출원 건수 | 45 |
| employment | 신규 고용 (천명) | 12.3 |
| gdp_growth | GDP 성장률 (%) | 2.8 |

**인과 구조 (분석 참고용):**
- 특허 = 0.4×R&D(t-5) + 0.2×기업지원(t-3) + 노이즈
- 고용 = 0.3×기업지원(t-2) + 0.1×세제혜택(t-1) + 노이즈

### 2.2 welfare_recipients.csv (복지 수혜자 데이터)

1,000명의 복지 수혜자 정보 (가상 데이터)

| 변수명 | 설명 | 예시 |
|--------|------|------|
| id | 수혜자 ID | 1, 2, 3, ... |
| age | 나이 | 45 |
| income | 연소득 (만원) | 1200 |
| region | 거주지역 (0=도시, 1=농촌) | 0 |
| benefit | 월 수급액 (만원) | 35 |

**데이터 특성:**
- 도시 거주자 비율: 약 70%
- 소득-수급액 상관: 약 -0.3 (부적 상관)

---

## 3. 과제 요구사항

### Task 1: 데이터 로드 및 탐색적 분석

1. 제공된 CSV 파일들을 로드하라
2. policy_timeseries.csv에 대해 다음을 수행하라:
   - 각 변수의 기초 통계량 (평균, 표준편차, 최소, 최대)
   - 정책 투입 변수(R&D, 기업지원, 세제혜택)와 결과 변수(특허, 고용, GDP) 간 상관행렬
   - 시계열 추세 시각화 (6개 변수를 2x3 서브플롯으로)
3. welfare_recipients.csv에 대해 다음을 수행하라:
   - 도시/농촌 비율 확인
   - 소득-수급액 상관계수 계산
   - 나이 분포 히스토그램

### Task 2: Transformer 기반 정책 효과 예측

다음 요구사항을 만족하는 Transformer 모델을 구현하라:

**모델 구조:**
- 입력: 과거 12개월 정책 데이터 (12 × 3변수 = 36차원)
- 출력: 다음 달 결과 변수 예측 (3차원: 특허, 고용, GDP)
- d_model: 32 (경량 모델)
- nhead: 4 (Attention head 수)
- num_layers: 2 (Transformer 층 수)

**학습 설정:**
- 에폭: 50
- 학습률: 0.001
- 손실함수: MSE (Mean Squared Error)
- 훈련/테스트 분할: 40:10 (처음 40개월 훈련, 마지막 10개월 테스트)

**평가 지표:**
- 각 결과 변수별 MAE (Mean Absolute Error)
- 각 결과 변수별 RMSE (Root Mean Squared Error)

### Task 3: CTGAN 합성 데이터 생성

welfare_recipients.csv를 사용하여 다음을 수행하라:

**CTGAN 학습:**
- 에폭: 100
- 범주형 변수: region
- 생성할 합성 데이터 수: 1,000명

**품질 평가 지표 계산:**
1. 통계적 유사도
   - 실제 vs 합성 데이터의 평균 나이 비교
   - 실제 vs 합성 데이터의 평균 소득 비교
   - 실제 vs 합성 데이터의 도시 비율 비교
   - 실제 vs 합성 데이터의 소득-수급액 상관계수 비교

2. 분포 매칭
   - 소득 변수에 대한 Wasserstein Distance 계산

### Task 4: 결과 종합 및 정책적 함의

1. Task 2의 Transformer 예측 결과를 다음 표 형식으로 정리하라:

   | 변수 | 실제 평균 | 예측 평균 | MAE | RMSE |
   |------|----------|----------|-----|------|
   | 특허 출원 | ? | ? | ? | ? |
   | 신규 고용 | ? | ? | ? | ? |
   | GDP 성장 | ? | ? | ? | ? |

2. Task 3의 CTGAN 품질 평가 결과를 다음 표 형식으로 정리하라:

   | 평가 항목 | 실제 데이터 | 합성 데이터 | 보존율 |
   |----------|------------|------------|--------|
   | 평균 나이 | ? | ? | ?% |
   | 평균 소득 | ? | ? | ?% |
   | 도시 비율 | ? | ? | ?% |
   | 소득-수급액 상관 | ? | ? | ?% |

3. 분석 결과를 바탕으로 다음 질문에 답하라:
   - Transformer 모델이 어떤 결과 변수를 가장 잘 예측하는가? 그 이유는 무엇인가?
   - CTGAN 합성 데이터가 정책 분석에 활용 가능한가? 어떤 한계가 있는가?

---

## 4. 제출물

**보고서** (report.pdf 또는 report.docx)
   - 4페이지 이내
   - 다음 구조를 따를 것:
     1. 서론 (분석 배경 및 목적)
     2. 방법론 (Transformer, CTGAN 모델 설명)
     3. 결과 (각 Task 결과 + 시각화)
     4. 결론 및 정책적 함의

---

## 5. 참고 자료

### 5.1 핵심 개념

**Transformer와 Self-Attention**
- Self-Attention 메커니즘: 시퀀스 내 모든 위치 간의 관계를 동시에 학습
- 장점: ARIMA/LSTM 대비 장기 의존성 학습에 유리
- 정책 분석 활용: R&D(t-5)→특허(t) 같은 시차 효과 자동 학습

**CTGAN (Conditional Tabular GAN)**
- Generator: 무작위 노이즈로부터 합성 데이터 생성
- Discriminator: 실제 데이터와 합성 데이터 구별
- 조건부 생성: 불균형 범주형 변수(도시 70%, 농촌 30%)도 효과적 학습
- 장점: 개인정보 보호하면서 통계적 특성 보존

**합성 데이터 품질 평가**
- 통계적 유사도: 평균, 분산, 상관계수 비교
- Wasserstein Distance: 두 분포 간 거리 측정 (낮을수록 좋음)
- 보존율: (1 - |실제-합성|/실제) × 100%

### 5.2 코드 힌트

```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from ctgan import CTGAN
from scipy import stats

# 데이터 로드 예시
policy_data = pd.read_csv('data/policy_timeseries.csv')
welfare_data = pd.read_csv('data/welfare_recipients.csv')

# Transformer 모델 골격
class PolicyTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=32, nhead=4, num_layers=2, output_dim=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # src: (batch, seq_len, input_dim)
        x = self.embedding(src)
        x = self.transformer_encoder(x)
        out = self.decoder(x[:, -1, :])  # 마지막 시점의 출력
        return out

# CTGAN 사용 예시
ctgan = CTGAN(epochs=100)
ctgan.fit(welfare_data, discrete_columns=['region'])
synthetic_data = ctgan.sample(1000)

# Wasserstein Distance 계산
w_distance = stats.wasserstein_distance(
    welfare_data['income'],
    synthetic_data['income']
)
```

---
