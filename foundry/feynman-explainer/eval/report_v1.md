# Feynman Explainer — Eval Report v1
*Generated: 2026-04-24 03:45*

## Aggregate Style Metrics

| Metric | Base Model | Fine-tuned | Δ | Target |
|--------|-----------|------------|---|--------|
| Feynman composite ↑ | 47.9 | 82.4 | +34.5 | higher=better |
| Flesch reading ease ↑ | 62.3 | 70.9 | +8.6 | 60–80 |
| Analogy density ↑ | 1.0 | 2.7 | +1.7 | >0.5 |
| Avg sentence length ↓ | 17.7 | 9.7 | -8.0 | 12–16 words |
| Avg word length ↓ | 4.1 | 4.0 | -0.1 | <5.0 chars |

## Response Comparisons (sample)

### Q: Why does a neural network need non-linear activation functions?

**Base model:**
> Imagine you're trying to build a simple model to predict how many points you'll score in a game based on how many hours you practice. You decide to use a linear function to predict your score. This function would look something like this: score = hours_practiced * 10 + 50. 
> 
> In this case, if you practice for 1 hour, you'd score 55 points, for 2 hours, 65 points, and so on. This is a linear relationship, meaning the increase in score is constant for every hour you practice.
> 
> But what if the relationship between practice and score isn't as straightforward? What if, instead of a constant increase, the score starts to increase more and more as you practice more? Maybe after 10 hours, you score 100 points, but after 20 hours, you score 200 points. This kind of relationship is non-linear.
> 
> Now, if you use a linear function to predict this, it won't capture the non-linear part of the relationship. It would still predict a constant increase, like 55 points for 1 hour, 65 for 2 hours, and so on, even though the actual relationship is different.
> 
> Neural networks are like this game prediction model. They're designed to learn complex relationships between inputs and outputs. If you use only linear activation functions, the network would be like using a linear function to predict your game score. It would struggle to capture the non-linear patterns in the data, like the sudden increase in score after 10 hours of practice.
> 
> Non-linear activation functions, like the sigmoid or ReLU functions, allow the network to model these complex, non-linear relationships.

**Fine-tuned:**
> Imagine you're trying to learn to identify dogs.  You have a bunch of pictures.  Some are golden retrievers, some are poodles, some are bulldogs.  You show these pictures to your friend, who's the expert.  They tell you, "This one is a golden retriever," and, "This one is a poodle."  Pretty easy for them, right?
> 
> Now, you want to build a computer model to do this identification. You give it a bunch of features. Maybe it's the size of the ears, the color of the fur, the snout shape. It builds up a bunch of numbers, a "feature vector," for each picture. Then it goes through a series of calculations, like adding up all those numbers and multiplying them by some weights. These weights are like the computer's learned connections to the dog's features.
> 
> Here's the thing. These early calculations, the ones that take your picture features and turn them into a single number, are like adding up all the numbers you used to describe a dog's ears. That number might be 50. Then you have to decide if that number is big enough to be a dog. You need a way to say, "Yes, this is a dog," or "No, this is a cat." That's where your non-linear activation function comes in.
> 
> Think of it like a little switch. This switch decides whether a particular feature, like the ears, is important enough to pass on to the next step. If the feature is *very* important, the switch lights up. If it's *not* important, the switch stays off. The key insight is that this switch introduces a *non-linear* relationship. It

*Composite: base=41.9 → fine-tuned=77.0 | Flesch: 52.9 → 66.5 | AvgSentLen: 16.6 → 11.7*

---

### Q: What is the difference between a parameter and a hyperparameter?

**Base model:**
> Let's think about this like we're playing with a toy car. Imagine you have a toy car that can go in different directions, and you want to make it go faster or slower. The speed at which the car moves is like a parameter. It's something you can adjust directly to change how the car behaves. You might change the speed by pressing a button or by adjusting a dial on the car.
> 
> Now, let's think about the car's engine. The engine is like a hyperparameter. It's something that affects how the car's speed can be adjusted. The engine can be powerful or weak, and it determines how fast the car can go. You might not be able to change the engine's power directly; you might need to buy a new car with a different engine. The engine's power is a setting that you can't change while the car is running, but you can choose it when you're picking out the car.
> 
> In the context of a computer program, parameters are like the speed of the toy car. You can change them to make the program do different things. Hyperparameters, on the other hand, are like the engine of the car. They're settings that you choose before you start running the program, and they can affect how the program behaves, but you can't change them while the program is running. For example, in a machine learning model, the learning rate (how fast the model learns from data) is a hyperparameter, while the weights and biases (which are adjusted during training) are parameters.
> 
> So, in summary, parameters are like the speed of the car you can adjust while driving, and hyperparameters are like the engine's power you choose before you start driving.

**Fine-tuned:**
> Imagine you're baking a cake.  You have the recipe itself.  This recipe is like the *data* for your cake.  It tells you how much flour, sugar, eggs, and so on you need.  This recipe is fixed.  You don't change it when you bake a different cake.  You just follow it.
> 
> Now, imagine you're not just baking one cake, but you're baking a whole batch of them. You want to make sure they're all the same, or at least close. To do this, you might adjust the oven temperature. You could set the temperature to 350 degrees. That's a *parameter*. It's a specific number, a thing you *change* to get the right oven setting for your batch.
> 
> But what if you want to bake a *special* cake, a bit different from the regular ones? You might add a little bit of vanilla extract. That vanilla extract is like a *hyperparameter*. It's not a direct ingredient for the cake itself. It's not something you directly measure or adjust within the baking process. It's something you *decide* to put in, or how much to put in, before you even start mixing the batter. The hyperparameter is the *settings* you set *before* you get started. It's the big knobs and dials you turn.
> 
> The key insight is that parameters are the knobs and levers you adjust *during* the process. Hyperparameters are the knobs and dials you adjust *before* you start the process. They are the fundamental settings that dictate the overall behavior of your model or your entire setup.

*Composite: base=48.1 → fine-tuned=71.2 | Flesch: 57.9 → 62.8 | AvgSentLen: 17.1 → 10.6*

---

### Q: Why does more training data usually beat a better algorithm?

**Base model:**
> Imagine you're trying to teach a robot to recognize cats in photos. You have two options: a simple algorithm that looks for big round eyes and a fluffy tail, or a more complex algorithm that can recognize cats in various poses and backgrounds.
> 
> Now, let's think about how you'd teach the robot. If you have a lot of photos of cats, the simple algorithm can learn to recognize cats pretty well. It might not be as sophisticated as the complex algorithm, but it can still spot cats in most photos, even if the cat is sitting on a chair or wearing a hat.
> 
> On the other hand, if you have very few photos of cats, the complex algorithm might struggle. It's like trying to teach a child to recognize cats by showing them only one or two photos. The child might not be able to generalize well and might not recognize cats in new situations.
> 
> So, more training data is like having more examples to teach the robot. It's like having a lot of photos of cats to show the robot. With enough examples, the robot can learn to recognize cats even if the simple algorithm is not as smart as the complex one.
> 
> In the end, having more training data is like having a bigger, more diverse classroom. It helps the robot (or in this case, the machine learning model) to learn better and generalize better to new situations. It's not that the simple algorithm is better, but that it can learn from the data you provide, and with more data, it can learn more effectively.
> 
> So, in a nutshell, more training data is like having a bigger, more diverse set of examples to teach from, which helps the model learn better, even if it's not as smart as a more complex algorithm.

**Fine-tuned:**
> Imagine you're trying to learn how to play the guitar.  You've got a really cool guitar.  It's got a fancy neck and a shiny body.  But you only play it for a tiny amount of time, say, an hour a week.  You're not very good with it.  Now, imagine you have a really *bad* guitar.  It's a cheap guitar, not much fun to play.  But you play it every single day, even if you're just fumbling around.  You get better, right?
> 
> Here's the thing. The guitar you have is like your algorithm. It's the *model*, the *architecture*, the *neurons* you're building. The fancy guitar is like a really clever algorithm, one with lots of fancy features, a complex structure, or a huge number of parameters. The better algorithm is just a really, really good recipe for making a guitar that plays well.
> 
> The training data is like the songs you practice. You can play a few songs on the fancy guitar, or you can play all the songs on the cheap guitar. The songs you play on the cheap guitar, even if you mess them up, they teach you new things about how the guitar works. You learn that pressing the "E" button on the neck doesn't always make a "C" sound.
> 
> The key insight is that the *quantity* of practice, the *amount* of training data, is often more important than the *quality* of the practice itself. If you have enough repetitions, even a simple method can start to look like a genius. Most people get confused because they think the fancy algorithm is the end result. It's not. It's just a

*Composite: base=37.3 → fine-tuned=77.9 | Flesch: 59.3 → 71.0 | AvgSentLen: 20.3 → 12.3*

---

## Verdict

✅ Strong style shift (+72.2% composite). Model card claims are justified.

**Retraining recommendation:** No retraining needed.

**Avg composite score:** base=47.9 → fine-tuned=82.4