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


## **1. Data**
1. More data
   
    &nbsp;&nbsp;
    이미지 분류에서 가장 중요한 것은 데이터의 양과 질이다. 또한 Pretrain된 모델에 추가로 분류할 데이터를 학습한다고 해도 추가되는 데이터의 양과 질은 보장되어야 한다. 따라서 학습을 위해 준비된 총 150장의 이미지 데이터는 충분한 양의 데이터가 되지못한다. 모델의 성능을 조금이라도 올리기 위해서 추가적인 데이터가 필요하다고 생각됐고, 150여 장의 이미지 데이터를 더 추가하여 **총 306장**의 이미지, Class 별로는 **formal: 98장, hiphop: 98장, vintage: 110장**의 이미지를 학습에 이용했다.
    <br>
    <br>

2. Preprocess
   
   &nbsp;&nbsp;
   수집된 패션 이미지 데이터에는 우리가 원하는 정보 (옷에 대한 정보) 외에도 학습에 필요없는 노이즈 정보 (배경에 대한 정보) 가 포함되어있다. 모델에 노이즈도 같이 학습하여 모델이 노이즈를 판단할 수 있도록 할수도 있지만, 임의로 학습전 노이즈를 제거하여 모델의 성능을 더 높일 수 있을 것이라고 생각했다. 따라서 ["removebg"](https://www.remove.bg/)에서 이미지의 배경 정보를 모두 지웠으며, 배경이 제거된 이미지의 예시는 아래와 같다.
   <br><br>
   <img src='./img/removebg_ex1.png'>
   <br><br>

3. Augmentation
    
    &nbsp;&nbsp;
    이미지 데이터가 적기 때문에 추가로 이미지를 수집해 줬지만, 그럼에도 불구하고 높은 정확도의 성능을 보여주기 위해서는 이미지 데이터의 양이 부족하다고 생각됐기 때문에 Augmentation을 이용해서 데이터의 양을 늘려보려고 했다. 우리가 수집한 데이터에 맞는 기법을 추려보니 **"Random Horizontal flip"** 과 **"Random Crop"** 을 사용하면 이미지에서 우리가 원하는 정보를 크게 훼손하지 않고 Augmentation을 할 수 있을 것이란 생각이 들었다. 따라서 각각의 기법에 대해서 모델의 성능에 어떤 영향을 주는지 확인해 봤다.
    <br><br>
    &nbsp;&nbsp;
    아래는 ResNet50 모델에서 파라미터 값 **learning rate = 0.00001, epoch = 150** 의 상황에서 **학습하는동안 보여준 최고 Test accuracy**를 각 기법 별 성능을 나타낸 것이다.
    <br>
    |-|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|avg|
    |-----|---|---|---|---|---|---|---|---|---|---|---|
    |Normal|0.79|0.89|0.87|0.84|0.87|0.85|0.84|0.82|0.84|0.87|0.85|
    |HorizontalFlip(0.5)|0.79|0.80|0.75|0.84|0.80|0.80|0.90|0.85|0.85|0.93|0.83|
    |Crop(224)|0.89|0.85|0.77|0.82|0.93|0.79|0.93|0.85|0.85|0.82|0.85|
    |Flip(0.5)+Crop(224)|0.84|0.82|0.85|0.77|0.85|0.82|0.77|0.77|0.85|0.77|0.81|
    <br>

    &nbsp;&nbsp;
    각 기법 별로 성능을 평가해본 결과, 예상과는 달리 Augmentation을 한 경우와 하지 않은 경우에 성능에서 큰 차이를 보여주지 못했다. 또한 50% 확률로 HorizontalFlip을 한 경우와 HorizontalFlip과 Crop을 같이한 경우에는 오히려 성능이 떨어지는 결과를 나타냈다. 이 결과를 바탕으로 이후 사용되는 **Augmentation은 224x224 사이즈 RandomCrop만 사용했다.** 
    <br><br>
    &nbsp;&nbsp;
    사용된 코드는 아래와 같다.   
      ~~~python
      train_transforms = transforms.Compose([transforms.Resize(img_size),
                                 transforms.RandomCrop(224),
                                 transforms.ToTensor(),
                                 ])
      test_transforms = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    ])
      ~~~
   <br>

## **2. Model**
1. ResNet

   &nbsp;&nbsp;
   이미지 분류를 위해서 먼저 ResNet을 사용했으며, pytorch에서 제공되는 ResNet50 모델에 Fully Connected layer를 추가하여 사용하였다. Pretrain된 모델을 사용하였기 때문에 준비된 이미지로 학습을 시킬 때에는 finetuning을 하지 않고 FC layer의 파라미터만 학습시켰다.
   <br><br>
   &nbsp;&nbsp;
   사용된 코드는 아래와 같다.
   ~~~python
   class Res(nn.Module):
      def __init__(self, num_cls = 3, pretrain=True, finetuning=True):
         super().__init__()
         self.model = models.resnet50(pretrained=pretrain)
         self.finetuning = finetuning
         if finetuning == False:
            for param in self.model.parameters():
                  param.requires_grad = False
         fc = []
         fc += [nn.Linear(self.model.fc.in_features, 512)]
         fc += [nn.ReLU()]
         fc += [nn.Linear(512, 128)]
         fc += [nn.ReLU()]
         fc += [nn.Linear(128, num_cls)]
         fc += [nn.LogSoftmax(dim=1)]
         self.model.fc = nn.Sequential(*fc)

      def forward(self, x):
         out = self.model.forward(x)
         return out

      def get_prams(self):
         if self.finetuning:
            return list(self.model.parameters()) + list(self.fc.parameters())
         else:
            return self.model.fc.parameters()
   ~~~
   <br>

   &nbsp;&nbsp;
   또한 학습할 이미지에 대한 ResNet의 적절한 learning rate 를 찾기 위해서 **learning rate가 0.00001일 때와 0.0001일 때** 모델의 성능을 비교해 보았다. 결과는 아래와 같다.
   <br>

    |-|1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|AVG|STD|
    |-----|---|---|---|---|---|---|---|---|---|---|---|---|
    |lr = 1E-5, epoch = 150|0.75|0.80|0.84|0.85|0.85|0.79|0.89|0.80|0.79|0.87|0.82|0.042|
    |lr = 1E-4, epoch = 100|0.89|0.77|0.82|0.82|0.92|0.82|0.80|0.79|0.82|0.75|0.82|0.049|
   
   &nbsp;&nbsp;
    각 learning rate 별로 성능을 평가해본 결과 정확도와 표준편차에서 의미있는 성능 차이를 나타내지 못했다. 따라서 **ResNet에 대해서**는  학습시간을 줄이기 위해서 epoch를 줄여도 일정 성능 이상을 보여주는 **learing rate = 0.0001을 사용**했다.
    <br><br>
    &nbsp;&nbsp;
    ResNet의 **train graph**는 아래와 같다.
    <br><br>
    <img src='./img/res_board.png'>
    
   <br><br>

2. DenseNet
3. MobileNet
4. Ensemble

## **3. Ensemble**
1. Three different Models
2. Three DenseNet

## **4. Result**