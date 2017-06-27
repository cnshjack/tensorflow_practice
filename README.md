# Tesorflow Stock Prediction

  Machine Learning is very hot now, lots of people discuss how to implement ML in people's life, like auto-driving. Some people suggest maybe we can use ML on financial area to make money, I believe there are many software engineer like me have a dream, which is maybe someday I can use ML to help myself make decision on investment, and then I will become very very rich!
  So now I choose Tensorflow as a ML tool, and choose some stocks, test whether we can predict stock price by using Tensorflow, if we can predict stock price accurately, well, it's time to say goodbye to your boss.
  
Environment:
  - Tensorflow 1.1.0
  - Python3
  - OSX

Stocks(Choosen from Taiwan Stock Exchange Corporation):
  - 台積電2330
  - 神基3005
  - 鴻海2317
 

### Test Result
The blue dot in graph is test data Y, and red dot is prediction data Y.

#### Stock 2330

![](https://github.com/evil0327/tensorflow_practice/blob/master/stock_predict_linear_regression/2330.png?raw=true)

#### Stock 3005

![](https://github.com/evil0327/tensorflow_practice/blob/master/stock_predict_linear_regression/3005.png?raw=true)

#### Stock 2317

![](https://github.com/evil0327/tensorflow_practice/blob/master/stock_predict_linear_regression/2317.png?raw=true)

The red dot and blue dot seems to be closely, but it is still hard to help make decision on investment, the average diffs in test data Y and predication data Y are 1.653(stock 2317), 1.6217(stock 3005) and 2.333(stock 2330). The test is for tomorrow's close price prediction, if we can't predict tomorrow price accurately, we don't know how to decide buy or sell stocks. But I think there is still possibilty to predict price if the input data X more correctlly, next time maybe input some technical indicator like KD, RSI, maybe the result will be different.


### Reference 
<https://github.com/LouisScorpio/datamining>

<https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg>
