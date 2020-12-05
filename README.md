# <div style="text-align: center"> **기초인공지능 PROJECT \#7** </div>
### <div style="text-align: right"> **2016142209 김태윤** </div>

**목차**

[**1. Data**](#1-data)
   1. More data
   2. Preprocess
   3. Augmentation
   
[**2. Model**](#2-model)
   1. ResNet
   2. DenseNet
   3. MobileNet
   4. Ensemble
   
[**3. Ensemble**](#3-ensemble)
   1. Three different models
   2. Three DenseNet
   
[**4. Result**](#4-result)

---

## **1. Data**
1. More data
   
    &nbsp;&nbsp;
    이미지 분류에서 가장 중요한 것은 데이터의 양과 질이다. 또한 Pretrain된 모델에 추가로 분류할 데이터를 학습한다고 해도 추가되는 데이터의 양과 질은 보장되어야 한다. 따라서 학습을 위해 준비된 총 150장의 이미지 데이터는 충분한 양의 데이터가 되지못한다. 모델의 성능을 조금이라도 올리기 위해서 추가적인 데이터가 필요하다고 생각됐고, 150여 장의 이미지 데이터를 더 추가하여 총 306장의 이미지, class 별로는 formal: 98장, hiphop: 98장, vintage: 110장의 이미지를 학습에 이용했다.
    <br>

2. Preprocess
   
   &nbsp;&nbsp;
   수집된 패션 이미지 데이터에는 우리가 원하는 정보 (옷에 대한 정보) 외에도 학습에 필요없는 노이즈 정보 (배경에 대한 정보) 가 포함되어있다. 모델에 노이즈도 같이 학습하여 모델이 노이즈를 판단할 수 있도록 할수도 있지만, 임의로 학습전 노이즈를 제거하여 모델의 성능을 더 높일 수 있을 것이라고 생각했다. 따라서 ["removebg"](https://www.remove.bg/)에서 이미지의 배경 정보를 모두 지웠으며, 배경이 제거된 이미지의 예시는 아래와 같다.
   <br>

3. Augmentation
    
    &nbsp;&nbsp;

    <br>

## **2. Model**
1. ResNet
2. DenseNet
3. MobileNet
4. Ensemble

## **3. Ensemble**
1. Three different Models
2. Three DenseNet

## **4. Result**