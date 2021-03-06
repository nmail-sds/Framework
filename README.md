# Framework v0.3
  
* 2018년 1월 23일 수정
* 김이한 (yihankim95@gmail.com)

* 삼성전략협업과제(반도체 생성 공정을 위한 딥러닝 기반 해석 가능한 이상 상태 검출 시스템) 프레임워크를 작성중입니다.
* 프로젝트를 위한 프레임워크는 설계한 모델을 `model/` 디렉터리에 넣으면 학습, 테스트가 가능하도록 합니다.

### 명령어 설명

`python main.py --dataset [dataset-name] --model [model-name] [--smote]`

* `--dataset` : 데이터셋 선택
* `--model` : 학습모델 선택
* `--smote` : 데이터셋 오버샘플링 기법 적용 여부

### main.py 실행 방법

`python main.py --dataset uci-secom --model linear-regression`

### 학습모델 작성 방법

* 모델의 이름은 `model/` 디렉토리의 코드 파일 이름으로 간주합니다.
* 모델 코드는 `class Model(object)`를 가지고 있어야 합니다.
* `Model()`은 `train`, `test` 메소드를 가져야 합니다.
  * `Model.train()` 함수는 data, label을 입력으로 받아 학습을 진행합니다.
    * 이외의 파라미터를 정의해야 할 경우 default 값을 추가해주셔야 합니다.
  * `Model.test()` 함수는 data를 받아 label을 리턴합니다.
* 이를 종합하면 아래와 같이 모델을 설계하여야 합니다.

```python
# model/mymodel.py 

class Model(object):
  def __init__(self):
    self.model = ... #
    ...
    return

  def train(self, data, label, epoch=100, ...):
    ... #train
    return 

  def test(self, data, ...):
    labels = ...
    return labels
```


### 현황*

* UCI-secom 데이터 기준 선형 회귀를 사용할 때 True를 놓치는 확률은 68%, False를 놓치는 확률은 4.8%입니다. 
