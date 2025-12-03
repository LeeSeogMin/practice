# AI 기반 정책분석방법론 기말과제

## 과제명: 에이전트 기반 모델링(ABM)을 활용한 지방자치단체 간 정책 확산 시뮬레이션

---

## 1. 과제 개요

### 1.1 배경

복잡계 이론에 따르면, 정책 시스템은 다수의 행위자들이 비선형적으로 상호작용하면서 거시적 수준에서 새로운 패턴을 창발시키는 복잡 적응계(Complex Adaptive System)의 특성을 보인다. 지방자치단체 간 정책 확산은 이러한 창발적 현상의 대표적 사례로, 개별 자치단체의 채택 결정이 네트워크를 통해 전파되면서 전국적 확산 패턴이 형성된다.

본 과제에서는 에이전트 기반 모델링(Agent-Based Modeling, ABM)을 활용하여 한국 지방자치단체 간 혁신 정책 확산 과정을 시뮬레이션하고, 네트워크 구조와 자치단체 특성이 확산 동학에 미치는 영향을 분석한다.

### 1.2 학습 목표

1. ABM의 핵심 개념(에이전트, 상호작용, 창발)을 이해하고 구현할 수 있다
2. 네트워크 구조가 정책 확산에 미치는 영향을 분석할 수 있다
3. 임계값 모델(Threshold Model)을 통해 티핑 포인트 현상을 설명할 수 있다
4. 시뮬레이션 결과를 바탕으로 정책적 함의를 도출할 수 있다

---

## 2. 제공 데이터

data 폴더에 다음 3개의 CSV 파일이 제공된다:

### 2.1 local_governments.csv (지방자치단체 데이터)

50개 지방자치단체의 특성 정보

| 변수명 | 설명 | 예시 |
|--------|------|------|
| id | 자치단체 고유 ID | 1, 2, 3, ... |
| name | 자치단체명 | 서울특별시, 부산광역시, ... |
| region | 권역 | 수도권, 영남권, 호남권, 충청권, 강원권, 제주권 |
| population | 인구수 | 9411000 |
| budget_billion | 예산규모(조원) | 42.3 |
| innovation_score | 혁신역량 점수 (0-1) | 0.92 |
| connectivity | 연결 중심성 (네트워크 연결 수) | 15 |
| initial_adopter | 초기 채택자 여부 (1=채택, 0=미채택) | 1 |

### 2.2 network_connections.csv (네트워크 연결 데이터)

자치단체 간 협력 네트워크 정보

| 변수명 | 설명 | 예시 |
|--------|------|------|
| source_id | 출발 노드 ID | 1 |
| target_id | 도착 노드 ID | 2 |
| connection_type | 연결 유형 | 행정협력, 권역협력, 지역내, 인접도시, 항공연결 |
| weight | 연결 강도 (0-1) | 0.8 |

### 2.3 policy_characteristics.csv (정책 특성 데이터)

분석 대상 정책의 특성 정보 (본 과제에서는 P001 스마트시티 통합플랫폼 정책을 분석)

| 변수명 | 설명 | 예시 |
|--------|------|------|
| policy_id | 정책 고유 ID | P001 |
| policy_name | 정책명 | 스마트시티 통합플랫폼 |
| complexity | 정책 복잡성 (0-1) | 0.85 |
| required_budget | 필요 예산(억원) | 150 |
| central_support | 중앙정부 지원율 (0-1) | 0.6 |
| adoption_benefit | 채택 시 기대 편익 (0-1) | 0.9 |

---

## 3. 과제 요구사항

### Task 1: 데이터 로드 및 네트워크 구축

1. 제공된 CSV 파일들을 로드하라
2. network_connections.csv를 활용하여 자치단체 간 네트워크를 구축하라
3. 네트워크의 기본 통계량을 계산하라:
   - 총 노드 수, 총 엣지 수
   - 평균 연결 차수(degree)
   - 네트워크 밀도(density)
   - 가장 연결이 많은 상위 5개 자치단체

### Task 2: 정책 확산 에이전트 모델 구현

다음 요구사항을 만족하는 PolicyAgent 클래스와 확산 시뮬레이션 함수를 구현하라:

**에이전트 속성:**
- id: 자치단체 ID
- name: 자치단체명
- adopted: 정책 채택 여부 (Boolean)
- threshold: 채택 임계값 (이웃 중 채택 비율이 이 값을 초과하면 채택)
- innovation_score: 혁신역량 점수
- neighbors: 이웃 에이전트 리스트

**채택 규칙:**
- 에이전트는 이웃 중 채택한 비율이 자신의 임계값을 초과할 때 정책을 채택한다
- 임계값은 1 - innovation_score로 계산한다 (혁신역량이 높을수록 낮은 임계값)
- 한 번 채택하면 철회하지 않는다

**시뮬레이션 요구사항:**
- 초기 채택자는 initial_adopter=1인 자치단체로 설정
- 최대 50단계(step)까지 시뮬레이션 수행
- 각 단계별 채택률과 신규 채택 수를 기록

### Task 3: 시뮬레이션 실행 및 결과 분석

1. 시뮬레이션을 실행하고 다음 결과를 도출하라:
   - 최종 채택률
   - 50% 채택 도달 시간 (티핑 포인트)
   - 90% 채택 도달 시간
   - 확산 패턴 (S자 곡선 여부)

2. 다음 표를 완성하라:

   | 지표 | 값 |
   |------|-----|
   | 초기 채택률 | ?% |
   | 최종 채택률 | ?% |
   | 50% 도달 단계 | ? |
   | 90% 도달 단계 | ? |
   | 총 확산 완료 단계 | ? |

3. 시간에 따른 채택률 변화를 시각화하라 (X축: 단계, Y축: 채택률)

### Task 4: 권역별 확산 패턴 분석

1. 권역(region)별로 확산 속도를 비교 분석하라
2. 어떤 권역이 가장 빠르게 확산되고, 어떤 권역이 가장 느린지 분석하라
3. 권역별 평균 혁신역량 점수와 확산 속도의 관계를 해석하라

---

## 4. 제출물

**보고서** (report.pdf 또는 report.docx)
   - 4페이지 이내
   - 다음 구조를 따를 것:
     1. 서론 (분석 배경 및 목적)
     2. 방법론 (ABM 모델 설계 설명)
     3. 결과 (각 Task 결과 + 시각화)
     4. 결론 및 정책적 함의

---

## 5. 참고 자료

### 5.1 핵심 개념

**에이전트 기반 모델링(ABM)**
- 개별 행위자(에이전트)의 속성, 행동 규칙, 상호작용을 명시적으로 모델링
- 미시적 행위로부터 거시적 패턴이 창발하는 과정을 시뮬레이션
- 방법론적 개인주의와 창발주의의 결합

**임계값 모델(Threshold Model)**
- Granovetter(1978)의 집합행동 임계값 이론
- 개인은 주변의 채택 비율이 자신의 임계값을 초과할 때 행동
- 티핑 포인트: 임계 질량(critical mass) 도달 시 확산 가속화

**네트워크 효과**
- 허브 노드: 연결이 많아 확산의 핵심 역할
- 브릿지 노드: 서로 다른 클러스터를 연결하여 확산 범위 확장
- 네트워크 밀도: 높을수록 빠른 확산, 낮으면 고립된 클러스터 형성

### 5.2 코드 힌트

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 로드 예시
local_gov = pd.read_csv('data/local_governments.csv')
network = pd.read_csv('data/network_connections.csv')

# 에이전트 클래스 골격
class PolicyAgent:
    def __init__(self, id, name, innovation_score, initial_adopter):
        self.id = id
        self.name = name
        self.adopted = (initial_adopter == 1)
        self.threshold = 1 - innovation_score  # 혁신역량이 높을수록 낮은 임계값
        self.neighbors = []

    def decide_adoption(self):
        """이웃의 채택 비율에 따라 채택 결정"""
        if self.adopted:
            return False  # 이미 채택함
        if len(self.neighbors) == 0:
            return False

        # 이웃 중 채택 비율 계산
        adoption_rate = sum(n.adopted for n in self.neighbors) / len(self.neighbors)

        # 임계값 초과 시 채택
        if adoption_rate > self.threshold:
            self.adopted = True
            return True  # 새로 채택함
        return False
```

---
