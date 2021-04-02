# **An introduction into Neural Style Transfer (NST)**
###### [These are the code examples](https://keras.io/examples/generative/neural_style_transfer/) written by François Chollet, author of Keras.  The base image by yours truly.

![gthbfnl1](https://user-images.githubusercontent.com/72076380/113143798-e0a41c80-922c-11eb-8d95-f13f29e669ff.jpg)



NST uses deep neural networks and allows you to transfer the style of 1 (style) image onto another (content) image. Pixel values are used as weights and biases to train the image to generate instead of making and training a model. 

It was first published in the paper "[A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf)" proposed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge in 2015.

Although it isn't "state of the art" and already used in Photoshop (neural filters) and mobile apps such as Copista, it is still interesting to see how it works and what can be created by changing parameters and hyper-parameters. 

## **▻ Excerpts from the research paper.**
**The key finding of the paper is that the representations of content and style in the Convolutional neural network are separable.**

The class of [Deep Neural Networks](https://arxiv.org/pdf/1605.07678.pdf?source=post_page---------------------------) that are most powerful in image processing tasks are called [Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53) (CNN or ConvNet). 

Convolutional comes from the mathematical operation on two functions (f and g) that produces a third function that expresses how the shape of one is modified by the other. It is a linear operation and [convolutional networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.](https://www.deeplearningbook.org/contents/convnets.html).

CNN = small computational units that process visual information hierarchically _feed-forward_. These networks care more about the higher layers such as shape and content (the more technical features).

Each layer of units = a collection of image filters, each of which extracts a certain feature from the input image.

Thus, the output of a given layer consists of so-called _feature maps_: differently filtered versions of the input image. 

<img width="780" alt="styleBank" src="https://user-images.githubusercontent.com/72076380/111777100-bce6eb00-88aa-11eb-9719-8a1fd3352655.png">

So a feature map is the output of 1 filter applied to the previous layer. A given filter is drawn across the entire previous layer, moved 1 pixel at a time. Each position results in an activation of the neuron and the output is collected in the feature map.



## ▻ VGG-Network
The name stems from the authors → Visual Geometry Group at Oxford. The results presented in the main text were generated on the basis of the VGG-Network, a CNN that rivals human performance on a common visual object recognition benchmark task. 

In the code example below [VGG19](https://iq.opengenus.org/vgg19-architecture/) is used. This CNN is 19 layers deep, and it classifies images. You can load a pre-trained version of the network trained on more than a million images from the ImageNet database. The pre-trained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224 by default.


# NST Components

###### [For the original contents](https://github.com/keras-team/keras-io/blob/master/examples/generative/md/neural_style_transfer.md). The programming language is Python and for (free) GPU we can use Google Colab and Kaggle Notebook.

**Table of contents:**

* Pre-processing utilities
* De-processing utilities
* Compute the style transfer loss
    1. Gram matrix
    2. style_loss function
    3. content_loss function
    4. total_variation_loss function
* Feature extraction model that retrieves intermediate activations of VGG19.
* Finally, here's the code that computes the style transfer loss.
* Add a tf.function decorator to loss & gradient computation.
* The training loop
* References
* Tools

## **▻ Pre-processing utilities**:
**Util function to open, resize and format pictures into appropriate tensors (multidimensional arrays).** 

```py
def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(
        image_path, target_size=(img_nrows, img_ncols)
    )
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)
```

## ▻ De-processing utilities:
**Util function to convert a tensor into a valid image.**

```py
def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
```

**Remove zero-centre by mean pixel.**

Zero-centring = shifting the values of the distribution to set the mean = 0. Removing zero-centre by mean pixel is a common practise to improve accuracy.

> [Caffe: will convert the images from RGB to BGR and then will zero-centre each colour channel with respect to the ImageNet dataset without scaling.](
https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py)

VGG19 is trained with each channel normalised by mean = [103.939, 116.779, 123.68]. These are the constants to use when working with this network.

```py
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
```

**'BGR'->'RGB'**  
R(ed)G(reen)B(lue) is the colour model for the sensing, representation, and display of images.  These are the primary colours of light (although light itself is an electromagnetic wave and colourless) and maximise the range perceived by the eyes and brain. Because we are working with screens that emit light instead of pigments, we use RGB.

The model formats the input image as batch size, channels, height and width as a NumPy-array in the form of BGR. VGG19 was trained using Caffe which uses OpenCV to load images and has BGR by default, therefore 'BGR'→'RGB' or x = x[:, :, ::-1]. 

Clip the interval edges to 0 and 255 otherwise we may pick values between −∞ and +∞.  Red, green and blue use 8 bits each, and they have integer values ranging from 0 to 255. 256³ = 16777216 possible colours. The date type = uint8 = Unsigned Integers of 8 bits (there are only 8 bits of information). Unsigned integers are integers without a "-" or "+" assigned to them. They are always non-negative (0 or positive) and we use them if we know that the outcome will always be non-negative.
```py
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x
```

## **▻ Compute the style transfer loss**
The loss/cost is the difference between the original and the generated image.  We can calculate this in different ways (MSE, Euclidean distance, etc.). By minimising the differences of the images, we are able to transfer style.

As commented in the code by Mr. Chollet.
>The "style loss" is designed to maintain the style of the reference image in the generated image.  It is based on the gram matrices (which capture style) of feature maps from the style reference image and from the generated image.
First, we need to define 4 utility functions:

**1. Gram matrix**: If we want to extract the style of an image, we need to compute the style-loss/cost. To do this we make use of the Gram matrix because it is the correlations between feature maps.

![1*HeCcGpmxWZFibgLiZCutag](https://user-images.githubusercontent.com/72076380/111915455-abead500-8a76-11eb-88cf-a50f8b6270c2.png)

Transpose: In linear algebra, the transpose of a matrix is an operator which flips a matrix over its diagonal; that is, it switches the row and column indices of the matrix A by producing another matrix, often denoted by Aᵀ.

Why does the Gram matrix represent artistic style? This is explained in the paper [Demystifying Neural Style Transfer](https://arxiv.org/pdf/1701.01036.pdf) proposed by Yanghao Li, Naiyan Wang, Jiaying Liu and Xiaodi Hou in 2017.

>We theoretically prove that matching the Gram matrices of the neural activations can be seen as minimising a specific Maximum Mean Discrepancy (MMD) [Gretton et Aal., 2012a]. This reveals that neural style transfer is intrinsically a process of distribution alignment of the neural activations between images.

When we are minimising the style loss, we are matching the distribution of features between two images. The gram matrix able to capture that distribution alignment of those feature maps in a given layer.

```py
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram
```


For the finite-dimensional real vector in R^n with the usual Euclidean dot product, the Gram matrix is simply G = V^T x V, where V is a matrix whose columns       are the vectors V(k).

Why the dot product? The dot product takes two equal-length sequences of numbers and returns a single number → a · b = |a| × |b| × cos(θ) or we can do → a · b = a<sub>x</sub> × b<sub>x</sub> + a<sub>y</sub> × b<sub>y 

Let’s take 2 flattened vectors representing features of the input space.  The dot product gives us information about the relation between them.  It shows us how similar they are.

* The lesser the product = the more different the learned features are.
* The greater the product = the more correlated the features are.



**2. The style_loss function:**
_**Which keeps the generated image close to the local textures of the style reference image.**_

**We take the Gram matrix of the activations at each layer in the network for both images. For a single image, the Gram matrix of its activations at a layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline) is given by:**

![formula](https://render.githubusercontent.com/render/math?math=G_%7Bij%7D%5El%20%3D%20%7BF_i%5El%7D%20%5Ccdot%20%7BF_j%5El%7D&mode=inline)

![formula](https://render.githubusercontent.com/render/math?math=G_%7Bij%7D%5E%7Bl%7D&mode=inline) = the dot product between the vectorised feature map ![formula](https://render.githubusercontent.com/render/math?math=i&mode=inline) and ![formula](https://render.githubusercontent.com/render/math?math=j&mode=inline) in layer  ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline). 

![formula](https://render.githubusercontent.com/render/math?math=F_%7Bi%7D%5E%7Bl%7D&mode=inline) = where ![formula](https://render.githubusercontent.com/render/math?math=F&mode=inline) is the representation of the generated image and where ![formula](https://render.githubusercontent.com/render/math?math=i&mode=inline) is the vectorised feature map in layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline). It is the activation for the ![formula](https://render.githubusercontent.com/render/math?math=i&mode=inline)-th feature map at layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline).

![formula](https://render.githubusercontent.com/render/math?math=F_%7Bj%7D%5E%7Bl%7D&mode=inline) = where ![formula](https://render.githubusercontent.com/render/math?math=F&mode=inline) the representation of the generated image and where ![formula](https://render.githubusercontent.com/render/math?math=j&mode=inline) is the vectorised feature map in layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline). It is the activation for the ![formula](https://render.githubusercontent.com/render/math?math=j&mode=inline)-th feature map at layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline).

**We can now define the style loss at a single layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline) as the Euclidean (L2) distance between the Gram matrices of the style and output images.**

![formula](https://render.githubusercontent.com/render/math?math=E_l%20%3D%20%5Cfrac%20%7B1%7D%7B4N_l%5E2%20M_l%5E2%7D%20%5Csum_%7Bi%2Cj%7D%20%28G_%7Bij%7D%5El%20-%20A_%7Bij%7D%5El%29%5E2&mode=inline)

![formula](https://render.githubusercontent.com/render/math?math=E_%7Bl%7D&mode=inline) = the Euclidean (L2) distance between the Gram matrices of the style and output images.

![formula](https://render.githubusercontent.com/render/math?math=N_%7Bl%7D&mode=inline) = a layer with ![formula](https://render.githubusercontent.com/render/math?math=N&mode=inline)![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline) distinct filters
has ![formula](https://render.githubusercontent.com/render/math?math=N&mode=inline)![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline) feature maps each of size ![formula](https://render.githubusercontent.com/render/math?math=M&mode=inline)![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline).

![formula](https://render.githubusercontent.com/render/math?math=M_%7Bl%7D&mode=inline) = the height times the width of the feature map.

![formula](https://render.githubusercontent.com/render/math?math=G_%7Bij%7D%5E%7Bl%7D&mode=inline) = the respective style representation in layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline) of the generated image ![formula](https://render.githubusercontent.com/render/math?math=G&mode=inline).

![formula](https://render.githubusercontent.com/render/math?math=A_%7Bij%7D%5E%7Bl%7D&mode=inline) = the respective style representations in layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline) of the original ![formula](https://render.githubusercontent.com/render/math?math=A&mode=inline).


**The last step is to compute the total style loss as a weighted sum of the style loss at each layer.**

![formula](https://render.githubusercontent.com/render/math?math=L_%7Bstyle%7D%28%5Cvec%20a%2C%5Cvec%20x%29%20%3D%20%5Csum_%7Bl%7D%20%7Bw_l%7D%7BE_l%7D&mode=inline)

![formula](https://render.githubusercontent.com/render/math?math=w_%7Bl%7D&mode=inline) = weighting factors of the contribution of each layer to the total loss.


```py
def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

```






**3. The content_loss function:**
_**Which keeps the high-level representation of the generated image close to that of the base image. An auxiliary loss function designed to maintain the "content" of the base image in the generated image.**_



So let p⃗ and x⃗ be the original image and the image that is generated and ![formula](https://render.githubusercontent.com/render/math?math=P%5El&mode=inline) and ![formula](https://render.githubusercontent.com/render/math?math=F%5El&mode=inline) their respective feature representation in layer ![formula](https://render.githubusercontent.com/render/math?math=l&mode=inline). We then define the squared-error loss between the two feature representations.

![formula](https://render.githubusercontent.com/render/math?math=L_%7Bcontent%7D%28%5Cvec%20p%2C%20%5Cvec%20x%2C%20l%29%20%3D%20%5Cfrac%20%7B1%7D%7B2%7D%20%5Csum_%7Bi%2Cj%7D%20%28F_%7Bij%7D%5El%20-%20P_%7Bij%7D%5El%29%5E2&mode=inline)


```py
def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))
```
   
**4. The total_variation_loss function:** 
_**This is a regularisation loss to improve the smoothness of the generated image by keeping it locally-coherent and ensuring spatial continuity.**_ This wasn’t used in the original paper. It was later added because optimisation to reduce only the style and content losses led to highly pixelated and noisy outputs. We are basically shifting pixel values. It is defined in the following function.

def total_variation_loss(x):   a = K.square( x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])    b = K.square( x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height -           1, 1:, :])    
return K.sum(K.pow(a + b, 1.25))

or in this case:

```py
def total_variation_loss(x):
    a = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :]
    )
    b = tf.square(
        x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(a + b, 1.25))
```

## ▻ Feature extraction model that retrieves intermediate activations of VGG19.

```py
#Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)
```

## ▻ Finally, here's the code that computes the style transfer loss.

```py
# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss
```
    
## ▻ Add a tf.function decorator to loss & gradient computation
To compile it and thus make it fast. A decorator enables us to add functionalities to objects without changing the structure. [TensorFlow provides the tf.GradientTape API for automatic differentiation; that is, computing the gradient of a computation with respect to some inputs, usually tf.Variables.](https://www.tensorflow.org/guide/autodiff)
```py
@tf.function
def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads
```
## ▻ The training loop
Repeatedly run vanilla gradient descent steps to minimise the loss and save the resulting image every 100 iterations.
We decay the learning rate by 0.96 every 100 steps.

Here, the optimisation algorithm is Stochastic Gradient Descent (SGD). Stochastic = having a random probability distribution or pattern that may be analysed statistically but may not be predicted precisely (oxford-languages). 

It goes through the entire data set and replaces the gradient with an estimate thereof.  The estimates are calculated from a randomly selected subset of that data. This optimiser has been in use since the sixties (of the last century). We are optimising pixel values instead of the parameters.

We can, of course, experiment with different optimisers to see which one gives the most desirable outcome.


```py
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

iterations = 4000
for i in range(1, iterations + 1):
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_reference_image
    )
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 100 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        img = deprocess_image(combination_image.numpy())
        fname = result_prefix + "_at_iteration_%d.png" % i
        keras.preprocessing.image.save_img(fname, img)
```

## References:

**Neural Style Transfer:**

* [A Neural Algorithm of Artistic Style by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge](https://arxiv.org/pdf/1508.06576.pdf)

* [Demystifying Neural Style Transfer by Yanghao Li, Naiyan Wang, Jiaying Liu and Xiaodi Hou](https://arxiv.org/pdf/1701.01036.pdf)

* [Neural style transfer - Keras, by François Chollet](https://github.com/keras-team/keras-io/blob/master/examples/generative/md/neural_style_transfer.md)

* [Neural style transfer | TensorFlow Core](https://www.tensorflow.org/tutorials/generative/style_transfer)

* [Neural Style Transfer - Wikipedia](https://en.wikipedia.org/wiki/Neural_Style_Transfer)


###### And considerably more!


**Convolutional Neural Networks:**

* [A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

* [How Do Convolutional Layers Work in Deep Learning Neural Networks?](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks)

* [Deep Neural Networks](https://arxiv.org/pdf/1605.07678.pdf?source=post_page---------------------------)

###### And considerably more!

**VGG19:**

* [ImageNet: VGGNet, ResNet, Inception, and Xception with Keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

* [Understanding the VGG19 Architecture](https://iq.opengenus.org/vgg19-architecture/)

* [VGG-19 Pre-trained Model for Keras](https://www.kaggle.com/keras/vgg19/home)

* [ImageNet](http://www.image-net.org)


###### And considerably more!



**Tensors:**

* [Introduction to Tensors](https://www.tensorflow.org/guide/tensor)

* [TensorFlow](https://www.tensorflow.org)

* [Keras](https://keras.io)




## Tools:

* GPU: Google Colab and Kaggle Notebook

* Equation editor: CodeCogs


###### Special thanks to François Chollet for providing us with this brilliantly written code!
