# FoundryML

**This is still a work in progress and by far one of my most challenging and exciting projects!**

FoundryML is a self-learning agentic system specifically crafted for Machine Learning automation. Give it a train set, initiate the process, then sit back and watch the *magic* unfold!

Here is a quick demo video of how FoundryML works and it's workflow. Dataset used in the demo gif: [Kaggle Titanic](https://www.kaggle.com/competitions/titanic)

User:
- Step 1: Upload a training set
- Step 2 (Optional): Describe the training set, ground truth label, task, anything.

That's it. No *really*, that's all you have to do. 

FoundryML on the other hand does a *little bit* more behind the curtains:

- **Preprocessing**:
    - Inspect training set data for missing and unique values, data types...
    - Build a preprocessing plan with instructions on which columns to drop, numeric and categorical column operations (eg. imputing, scaling), as well as feature engineering operations.
    - Execute preprocessing plan, transforming and feature engineering the training set.

- **Training**:
    - Create a training plan using the general plan, preprocessing specifications, ground truth label and task type. Training plan consists of instructions for tasks such as train test split, cross validation, models to use, param grids for each model to be tested.
    - Execute training plan and perform GridSearchCV on every single model with the provided param grids in the training plan. 
    - Validate against a validation set, compare scores and performance of models. Save best one.

- **Packaging**:
    - Package the full pipeline process including preprocessing and feature engineering as a `.pkl` file, ready for user to download.

![](./img/FoundryMLDemo2.gif)

## FoundryML provides the following features:
- ✅ Full Machine Learning process from data inspection, to preprocessing and feature engineer, to training with different models and optimizing. 

- ❌ (To be implemented) History of previous tasks along with their traces and final status (eg. fail, success) inside a DB. FoundryML can revisit these past tasks, compare current task, retrieve relevant useful information and apply in it in it's current task. 

- ❌ (To be implemented) Self-learning is achievable through the DB of previous tasks. FoundryML will go through failed tasks, troubleshoot what went wrong, come up with a new plan and execute it, and record new findings. 

