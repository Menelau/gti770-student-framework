[![GitHub issues](https://img.shields.io/github/issues/pldelisle/machine-learning-docker-environment.svg)](https://github.com/pldelisle/machine-learning-docker-environment/issues) [![GitHub stars](https://img.shields.io/github/stars/pldelisle/machine-learning-docker-environment.svg)](https://github.com/pldelisle/machine-learning-docker-environment/) [![GitHub forks](https://img.shields.io/github/forks/pldelisle/machine-learning-docker-environment.svg)](https://github.com/pldelisle//machine-learning-docker-environment/network) ![GitHub license](https://img.shields.io/badge/license-MIT-yellow.svg) 

# Machine Learning Docker Environment 
<img src="images/code.png" width="96" height="96" vertical-align="bottom">

### Introduction

This is the Git repository for the source code of the framework used for realizing GTI770-Systèmes intelligents et apprentissage machine course's labs.

This code is only the framework and is incomplete to let the student explore several machine learning algorithms. It is used jointly with multiple datasets,
such as [GalaxyZoo](https://www.galaxyzoo.org), [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/) and [Spambase Data Set](https://archive.ics.uci.edu/ml/datasets/spambase).

Students need to complete with their own code to solve classification problems automatically using different machine learning algorithms such as KNN, Naive Bayes, SVM, Neural Networks and Decision tree/Random Forests. 

This framework has many dependencies, such as OpenCV 3.x.x, scikit-learn and TensorFlow. A best practice consists of running the code using a Docker environment built with all dependencies : [Machine Learning Docker Environment](https://github.com/pldelisle/machine-learning-environment).
This framework has some code that can be GPU-accelerated using the GPU-enabled Docker environment and an NVIDIA GPU.

### Quick references

* Maintained by: 

	[Pierre-Luc Delisle](https://github.com/pldelisle) 

* Where to file issues: 
	
	[Github issues](https://github.com/pldelisle/gti770-student-framework/issues)

* Supported architectures:

	`[amd64]` `[amd64-nvidia]`


### Minimum requirements

* 1.5 GB free hard disk space
* A minimum of a 4-core, 4-thread x86 CPU. 
* A minimum of 8 GB of RAM, 16 GB or more is highly recommended.

### Notes

OpenCV and TensorFlow, can be GPU-accelerated using NVIDIA GPU. 

The OpenCV version required to run this code is OpenCV 3.3.x. OpenCV must be compiled for Python3.


### Usage

#### Getting started

Ensure you have an environment variable according to this table : 

| With Docker                                     || Without Docker                              ||  
|:--------------:|:-------------------------------:|:-----------------:|:------------------------:|  
| Name           | Value                           | Name              | Value                    |  
| VIRTUAL_ENV    | /home/ubuntu/ml_venv/project    | VIRTUAL_ENV       |  /home/ubuntu/ml_venv/   |  


To launch the script :
`cd core`  
`python3 main.py`


### How to contribute ?
- [X] Create a branch by feature and/or bug fix
- [X] Get the code
- [X] Commit and push
- [X] Create a pull request

#### Branch naming

##### Feature branch
> feature/ [Short feature description] [Issue number]

##### Bug branch
> fix/ [Short fix description] [Issue number]

##### Commits syntax:

##### Adding code:
> \+ Added [Short Description] [Issue Number]

##### Deleting code:
> \- Deleted [Short Description] [Issue Number]

##### Modifying code:
> \* Changed [Short Description] [Issue Number]


### Credits

<div>Icons made by <a href="https://www.flaticon.com/authors/smashicons" title="Smashicons">Smashicons</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a></div>
