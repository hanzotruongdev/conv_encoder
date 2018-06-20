

# Conv-encoder
It is final project of the class "Advanced multimedia image processing" of my master course in Dongguk Univ.

Conv-encoder is `Convolutional Auto Encoder` which reduces the dimension of image from 28x28 to 12x12 but still keeps almost the important features of the image.
Conv-encoder is a machine-learning-based component which uses example for training. For example, I use `MNIST handwrittend digits dataset` to train the mmodel with 1000 epoch.
The dimensionality reduction of conv-encoder, it is meaningful for apply in other models to reduce the complexity and computation cost but still keeps the high accuracy of model.

# Usage #

1. Clone project to your PC
2. Move to project folder
```sh
$cd conv-encoder
```

3. Test with pre-trained model:
```sh
$python3 main.py test
```

Check `output` folder for result

4. Retrain the model:
```sh
$python3 main.py train
```

The model will be retrained and store in folder `models`. In order to test new model, return to step 3.


# Example output
![alt text](https://raw.githubusercontent.com/noitq/conv-encoder/master/example1.jpg)
![alt text](https://raw.githubusercontent.com/noitq/conv-encoder/master/example2.jpg)
![alt text](https://raw.githubusercontent.com/noitq/conv-encoder/master/example3.jpg)
![alt text](https://raw.githubusercontent.com/noitq/conv-encoder/master/example4.jpg)