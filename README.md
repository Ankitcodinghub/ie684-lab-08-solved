# ie684-lab-08-solved
**TO GET THIS SOLUTION VISIT:** [IE684 Lab 08 Solved](https://www.ankitcodinghub.com/product/ie684-lab-08-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;97620&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;IE684 Lab 08 Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Binary Classification Problem

In the last lab, you might have recognized that decomposing a problem might help in coming up with optimization procedures to handle large data. In this lab, we will continue with this theme and try to develop procedures which are scalable and achieve reasonably accurate solutions.

Here, we will consider a different problem, namely the binary (or two-class) classification problem in machine learning. The problem is of the following form. For a data set D = {(xi , yi )}ni=1 where xi ∈X ⊆Rd,yi ∈{+1,−1},wesolve:

</div>
</div>
<div class="layoutArea">
<div class="column">
λ 1􏰂n min f(w) = ∥w∥2 +

</div>
</div>
<div class="layoutArea">
<div class="column">
L(yi, w⊤xi). (1) Note that we intend to learn a classification rule h : X → {+1, −1} by solving the problem (1).

We will use the following prediction rule for a test sample xˆ:

h(xˆ) = sign(w⊤xˆ). (2)

We will consider the following loss functions:

<ul>
<li>􏰁 &nbsp;Lh(yi,w⊤xi)=max{0,1−yiw⊤xi} (hinge)</li>
<li>􏰁 &nbsp;Ll(yi, w⊤xi) = log(1 + exp(−yiw⊤xi)) (logistic)</li>
<li>􏰁 &nbsp;Lsh(yi, w⊤xi) = (max{0, 1 − yiw⊤xi})2. (squared hinge)</li>
<li>􏰁 &nbsp;Exercise 0: [R] For an example (x, y) ∈ X ×Y , assume z = yw⊤x. Then, note that the loss functions Lh, Ll and Lsh can be equivalently written as Gh(z), Gl(z), Gsh(z). Write the loss functions Gh(z),Gl(z) and Gsh(z) as functions of z. Plot these loss functions Gh(z),Gl(z) and Gsh(z) where z takes values on the real line [−∞, ∞]. Distinguish the loss functions using different colors. Comment on the behavior of respective loss functions with respect to z.</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
w∈Rd 2n

</div>
<div class="column">
i=1

</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 08 09-March-2022

Exercise 1: Data Preparation

1. Use the following code snippet. Load the iris dataset from scikit-learn package using the following code. We will load the features into the matrix A such that the i-th row of A will contain the features of i-th sample. The label vector will be loaded into y.

<ol>
<li>(a) &nbsp;[R] Check the number of classes C and the class label values in iris data. Check if the class labels are from the set {0,1,…,C −1} or if they are from the set {1,2,…,C}.</li>
<li>(b) &nbsp;When loading the labels into y do the following:
<ul>
<li>􏰁 &nbsp;If the class labels are from the set {0,1,…,C −1} convert classes 0,2,3,…,C −1 to −1.</li>
<li>􏰁 &nbsp;If the class labels are from the set {1,2,…,C} convert classes 2,3,…,C to −1. Thus, you will have class labels eventually belonging to the set {+1, −1}.</li>
</ul>
</li>
<li>(c) &nbsp;Note that a shuffled index array indexarr is used in the code. Use this index array to partition the data and labels into train and test splits. In particular, use the first 80% of the indices to create the training data and labels. Use the remaining 20% to create the test data and labels. Store them in the variables train data, train label, test data, test label.
<pre>          import numpy as np
</pre>
<pre>          #we will load the iris data from scikit-learn package
</pre>
<pre>          from sklearn.datasets import load_iris
          iris = load_iris()
          #check the shape of iris data
          print(iris.data.shape)
</pre>
A = iris.data

<pre>          #check the shape of iris target
</pre>
<pre>          print(iris.target.shape)
</pre>
<pre>          #How many labels does iris data have?
          #C=num_of_classes
          #print(C)
          n = iris.data.shape[0] #Number of data points
          d = iris.data.shape[1] #Dimension of data points
</pre>
<pre>          #In the following code, we create a nx1 vector of target labels
</pre>
<pre>          y = 1.0*np.ones([A.shape[0],])
          for i in range(iris.target.shape[0]):
</pre>
<pre>             # y[i] = ???? # Convert class labels that are not 1 into -1
</pre>
<pre>          #Create an index array
          indexarr = np.arange(n) #index array
          np.random.shuffle(indexarr) #shuffle the indices
          #print(indexarr) #check indexarr after shuffling
</pre>
<pre>          #Use the first 80% of indexarr to create the train data and the remaining 20% to
              create the test data
</pre>
<pre>          #train_data = ????
          #train_label = ????
          #test_data = ????
          #test_label = ????
</pre>
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab Lab 08

(d)

(e)

</div>
<div class="column">
09-March-2022

</div>
</div>
<div class="layoutArea">
<div class="column">
Write a python function which implements the prediction rule in eqn. (2). Use the following code template.

<pre>def predict(w,x):
  #return ???
</pre>
Write a python function which takes as input the model parameter w, data features and labels and returns the accuracy on the data. (Use the predict function).

<pre>def compute_accuracy(data,labels,model_w):
  #Use predict function defined above
  #return ???
</pre>
</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab Lab 08

Exercise 2: An Optimization Algorithm

1. Note that problem (1) can be written as

</div>
<div class="column">
09-March-2022

</div>
</div>
<div class="layoutArea">
<div class="column">
min f(w) = min 􏰂 fi(w).

ww

i=1

[R] Find an appropriate choice of fi(w).

2. Consider the loss function Lh. Write a python module to compute the loss function Lh.

<pre>   def compute_loss_h(w,x,y):
     #return ???
</pre>
</div>
<div class="column">
(3)

</div>
</div>
<div class="layoutArea">
<div class="column">
<ol start="3">
<li>Write a python routine to compute the objective function value. Use the compute loss function.
<pre>   def compute_objfnval(data,labels,model_w):
     #return ???
</pre>
</li>
<li>Write an expression to compute the gradient (or sub-gradient) of fi(w) for the loss function Lh. Denote the (sub-)gradient by gi(w) = ∇wfi(w). Define a python function to compute the gradient.
<pre>   def compute_grad_loss_h(x,y,model_w):
     #return ???
</pre>
</li>
<li>Write an optimization algorithm where you pass through the training samples one by one and do the (sub-)gradient updates for each sample. Recall that this is similar to ALG-LAB8. Use the following template.
def OPT1(data,label,lambda, num_epochs): t=1

<pre>     #initialize w
     #w = ???
     arr = np.arange(data.shape[0])
     for epoch in range(num_epochs):
</pre>
<pre>      np.random.shuffle(arr) #shuffle every epoch
      for i in np.nditer(arr): #Pass through the data points
</pre>
<pre>        # step = ???
        # Update w using w &lt;- w - step * g_i (w)
        t = t+1
        if t&gt;1e4:
</pre>
t=1 return w
</li>
<li>In OPT1, use num epochs=1000, step=1t . For each λ ∈ {10−3,10−2,0.1,1,10}, perform the following tasks:
(a) [R] Plot the objective function value in every epoch. Use different colors for different λ values.

(b) [R] Plot the test set accuracy in every epoch. Use different colors for different λ values. 5
</li>
</ol>
</div>
</div>
<div class="layoutArea">
<div class="column">
n

</div>
</div>
</div>
<div class="page" title="Page 6">
<div class="layoutArea">
<div class="column">
IE684, IEOR Lab

Lab 08 09-March-2022

(c) [R] Plot the train set accuracy in every epoch. Use different colors for different λ values. (d) [R] Tabulate the final test set accuracy and train set accuracy for each λ value.

(e) [R] Explain your observations.

7. [R] Note that in OPT1, a fixed number of epochs is used. Can you think of some other suitable stopping criterion for terminating OPT1? Implement your stopping criterion and check how it differs from the one in OPT1. Use step=1t and λ which achieved the best test set accuracy in the previous experiment.

8. [R] Repeat the experiments (with num epochs=1000 and with your modified stopping crite- rion) for different loss functions Ll and Lsh. Explain your observations.

</div>
</div>
<div class="layoutArea">
<div class="column">
6

</div>
</div>
</div>
