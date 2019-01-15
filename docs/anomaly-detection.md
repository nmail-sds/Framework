# 이상 감지(Anomaly Detection) 알고리즘

* 참고 : [Oracle Data Science](https://www.datascience.com/blog/python-anomaly-detection)

## 소개
* **아웃라이어 패턴을 찾아내는 기술
* 사업, 시스템 상태 모니터링에서의 침해 감지, 신용카드의 도용을 감지하는 등의 응용 분야가 있음

## Anomaly란 무엇인가*
* 크게 세 가지로 분류 가능한데:
  * Point anomaly: 나머지 데이터로부터 멀리 떨어져 있는 데이터. 예를 들면 신용카드 결제금액을 바탕으로 사기를 분류하는 기법이 있다.
  * Contextual anomaly: 이상이 특정 맥락을 가짐. 시계열 데이터(Time-series)에서 자주 발생. 
  * Collective anomaly: 기존 데이터 항목의 집합이 이상 감지에 도움을 줄 때. 사용 경력이 없는 사용자가 로컬호스트에 접근할 때 이를 잠재적 사이버공격으로 보는 것이 예이다.

* 이상 감지는 noise removal, novelty detection과 유사한 면이 있다. 
  * Novelty detection은 학습 데이터에 나타나지 않은 패턴을 감지하는 것을 말한다.
  * Noise removal(NR)은 의미 있는 신호에 포함된 잡음을 제거하는 과정이다.


## 이상 감지 기술

#### 간단한 통계적 기법

* 가장 간단한 접근 방법은 데이터 분포의 성질들(평균, 표준편차, 최빈값, 중간값 등)로부터 데이터 값들을 표시하는 것이다.
* 연속한 n개의 원소를 평균내는 방법이 있으며, 이를 로우패스 필터라고 정의한다.

#### 챌린지

* ㅁㅇㄴㄻㄴ어라ㅣ

## 기계학습을 이용한 접근

#### 밀도기반 이상감지(Density-based Anomaly Detection)

* KNN
* Local outlier factor(LOF)

#### 군집기반 이상감지(Cluster-based Anomaly Detection)

* K-means

#### 서포트 벡터 머신(SVM, Suppoer Vector Machine) 기반 이상감지


## 코드(생략) 
