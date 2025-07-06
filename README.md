# Naive Bayes Model for Spam Classification

## Bayes Theorem

To understand Naive Bayes, we must first look at [Bayes' theorem](https://www.probabilitycourse.com/chapter1/1_4_3_bayes_rule.php):

### $$P(A|B) = \frac{P(A)⋅P(B|A)}{P(B)}$$

**Where :**\
$P(B)$ is the probablity of B occurring and\
$P(A|B)$ is the probablity of A occuring given that B has already occurred\
$P(A)$ is the probablity of A occurring and\
$P(B|A)$ is the probablity of B occuring given that A has already occurred

</br>

**In essence**, with Bayes' theorem we can find out the posterior probablity of an event B occuring given that event A has occurred if we know the proability of event A occuring, event B occurring and event A occurring given that event B occurred.

**For example**, in spam classification we know the probability of an event **S** being a spam mail and the probablity of the event **B** that the word Bitcoin is in a mail. And forutnately, we also know the probability of the event B given S i.e that if the mail is a spam the word 'Bitcoin' is there or not.

These are all probability that you can find if you've got a mail dataset which has an label of the mail being spam or not.

With this data what we would love to do is predict if the new email that we recieve is spam or not. And that can be done using Bayes' theorem. We can find the probablity that the email being spam given that the word 'Bitcoin' is in the email using all of the probablitiies that we've collected before.

### $$P(S|B) = \frac{P(S)⋅P(B|S)}{P(B)}$$

**Where :**\
$P(B)$ is the probablity of the word 'Bitcoin' being on the mail\
$P(S|B)$ is the probablity of the email being spam  given that 'Bitcoin' is in the mail\
$P(S)$ is the probablity of the mail being spam\
$P(B|S)$ is the probablity of 'Bitcoin' being on the mail given that the email is a spam

## Full Bayes theorem

### The Bayes' theorem can be further extended:

![Full Bayes' Theorem.png](data:image/png;)

$$ P(A|B) = \frac{P(A)⋅P(B|A)}{P(B)} \tag{1}$$

From the venn diagram given above we can see that the set B can be represented as the union of two disjoint sets:
- $A∩B$
-  $B∩\tilde{A}$

$$
P(B) = P\left( (A \cap B) \cup (B \cap \tilde{A}) \right) \tag{2}
$$

The [axioms of probability](https://www.probabilitycourse.com/chapter1/1_3_2_probability.php) state that if $A_1, A_2, A_3, \ldots$ are disjoint events (sets), then

$$
P(A_1 \cup A_2 \cup A_3 \cup \ldots) = P(A_1) + P(A_2) + P(A_3) + \ldots\tag{3}
$$

<br/>

Since $A∩B$ and $B∩\tilde{A}$ are disjoint sets (they don't intersect) ; from $(2)$ and $(3)$:

$$P(B) = P(A \cap B)  +  P(B \cap \tilde{A}) \tag{4}$$

Now from eqn $(1)$ and $(4)$:

$$P(A|B) = \frac{P(A)⋅P(B|A)}{P(A \cap B)  +  P(B \cap \tilde{A})} \tag{5}$$

Now using [conditional probablity](https://www.probabilitycourse.com/chapter1/1_4_0_conditional_probability.php) for the denominatior:


$$P(A|B) = \frac{P(A)⋅P(B|A)}{P(A)⋅P(B|A)  +  P(\tilde{A})⋅P(B| \tilde{A})} \tag{6}$$

### which is the **Full Bayes' Theorem**
----

## Naive Bayes

<p>

 The key idea behind Naive Bayes is the assumption that each predictor is independent of the others. For example, the word <i>"win"</i> is considered independent of the word <i>"million"</i>. This is a strong (and often unrealistic) assumption, which is why the method is called "naive."

</p>

<p> This assumption allows each event in the predictor to be treated as independent, simplifying the model training process significantly. Without it, building a conditional model for every word in a spam message would be extremely difficult. Despite its strong assumption, Naive Bayes often performs surprisingly well and has historically been used in spam filters. </p>


## Spam classification using Naive Bayes

Let us consider words $w_1, w_2, \ldots, w_n$ that could be in a message. Let $x_i$ be the event that the word $w_i$ is in the message given where $i = 1,2,3,\ldots,n$.

Our goal is to find the probablity of the words in a message bein spam ie, the message being spam given the following words (bayes' terminnology). What we want to find is:

$$P(S|x_1 , x_2\ldots, x_n ) \tag{7}$$

**Where:**\
S is the event that the message is a spam\
$X_i$ be the event that the word $w_i$ is in the message, $i \in \{1,2,\ldots, n\}$

To find this we'll first need to compute (Bayes' theorem):

$$P(x_1 , x_2\ldots, x_n|S )\tag{8}$$

***Note:*** $P(A\cap B) = P(A,B) = P(A\ and\ B)$

This could've been an extremely difficult computation had we not made our assumption that the events are independent of each other ( The word *win* has nothing to do with the word *MILLION* ).

<br>

We know that for [independent events](https://www.probabilitycourse.com/chapter1/1_4_1_independence.php):\
$P(A\cap B) = P(A)⋅P(B)$

<br>

Using the given formula for independent events in $(2)$ :

$$P(x_1 , x_2\ldots, x_n|S ) = P(x_1|S)\cdot P(x_2|S)\ldots P(x_n|S) \tag{9}$$

Similarly,

$$P(x_1 , x_2\ldots, x_n|\tilde{S} ) = P(x_1|\\tilde{S} )\cdot P(x_2|\tilde{S})\ldots P(x_n|\tilde{S} ) \tag{10}$$

**Where**:\
S is the event that the message is *NOT* spam

---

## Important Considerations

In practise however, we want to avoid multiplying these probabilities to prevent [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow) which is the computer's inablility to compute well with floating point numbers too close to zero. If we were to multiply them, we could likely get a probablity zero which would mean **wrong** classification.
<br>

Enter logarithm to the rescue.
We know:
$$log(a) + log(b) = log(ab) \tag{11}$$
and
$$ exp(log(a)) = a \tag{12}$$

For proofs see: [Logarithmic properties](https://www.khanacademy.org/math/algebra2/x2ec2f6f830c9fb89:logs/x2ec2f6f830c9fb89:log-prop/a/justifying-the-logarithm-properties)

<br>


$$ P(x_1|\tilde{S})\cdot P(x_2|\tilde{S};)\ldots P(x_n|\tilde{S} )  = exp(log[(P(x_1|\tilde{S} )\cdot P(x_2|\tilde{S} )\ldots P(x_n|\tilde{S} )])$$
$$ P(x_1|\tilde{S} )\cdot P(x_2|\tilde{S} )\ldots P(x_n|\tilde{S} )  =  exp[log(P(x_1|\tilde{S} )) + log(P(x_2|\tilde{S} )) + \ldots +log(P(x_n|\tilde{S} ))] \tag{13}$$



## Naive Bayes Model

Finally, we've come to our model. This is where we'll use our Bayes' theorem to predict if a model is spam or not.

Let $X$  be all the events such that $X = x_1,x_2,\ldots,x_n$


#### From Bayes theorem:

$$P(S|X ) = \frac{P(S)\cdot P(X |S)}{P(S)\cdot P(X|S) + P(\tilde{S})\cdot P(X|\tilde{S})} \tag{14}$$

$$ = \frac{1}{1 + \frac{P(\tilde{S})\cdot P(X |\tilde{S})}{P(S)\cdot P(X |S)}}$$

Remeber that we use logarithm to prevent underflow:

$$ = \frac{1}{1 + \frac{e^{(log(P(\tilde{S})\cdot P(X = x_i|\tilde{S}))}}{e^{(log(P(S)\cdot P(X = x_i|S))}}}$$

From $(13)$


$$  = \frac {1}{1 + \frac{e^{logP(\tilde{S})+logP(X|\tilde{S})}}{e^{logP(S)+logP(X|S)}}}$$

$$ = \frac{1}{1 + e ^{logP(\tilde{S})+log(X |\tilde{S})-logP(S)-log(X|S)}}\tag{15}$$
<br>


#### To find the proabibility for an email being spam we'll just use the formula for every single word in the email.

To do this we can use eqn $(9), (10)$ and replace them in $(15)$

$$log(P(x_1 , x_2\ldots, x_n|S )) = \sum_{i = 1}^nlog(P(x_i|S)) \tag{16}$$


$$log(P(x_1 , x_2\ldots, x_n|\tilde{S} )) = \sum_{i = 1}^nlog(P(x_i | \tilde{S})) \tag{17}$$

Finally we get to the final equation for Naive Bayes' Model:

$$ = \frac{1}{1 + e ^{logP(\tilde{S})+\sum_{i = 1}^nlog(P(x_i | \tilde{S}))-logP(S)- \sum_{i = 1}^nlog(P(x_i|S))}}\tag{18}$$


## Laplace Smoothing

Are we still not done yet? Fortunately this is the last step of our Naive Bayes' Model.
Let's say that we have a word 'mathematics' that was found only in non spam email during our training process.
So $P(x | S ) = 0$, where $x$ is the event that the word 'mathematics' was in the email. \

<br>

Since we're multiplying probablities for classification,  when prediciting the class of an unseen data, if we encounter the word 'mathematics' our model is going to predict the probablity of the mail being spam as 0 even though the email could actually be spam.

<br>

To prevent this issue we use **Laplace smoothing**.
While calculating probability for a word being spam we'll be using a pseudocount $k$ and estimate the probablity for the work $w_i$.

$$P(x_i | S) = \frac{k + N\cdot w_i}{n⋅k + S_n}\tag{19}$$

**Where:**\
$N$ is the number of spam email containing the word $w_i$\
$k$ is the psuedocount, set at random but typically 1\
$n$ is the number of output variables. Since this the binary classification it will be 2\
$S_n$ is the total number of spam emails

In this way even if an word does not appear in spam mail while training it will have a very small probablity 
### $\frac{k + N\cdot w_i}{2⋅k + S_n}$
and not 0. This will prevent the zero probablity error for spam mail containing words that were all present in non-spam email data during our model training process.

<br>

----
----
