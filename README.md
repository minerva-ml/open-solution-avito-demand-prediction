# Avito Demand Prediction Challenge: open solution

This is an open solution to the [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction).

## The goal
Create (entirely) open solution to this competition. We are opening not only the code, but also the process of creating it. Rules are simple:
* Clean code and extendable solution are - in the long run - much better than current public LB position
* This solution should - by itself - establish solid benchmark, as well as provide good base for your custom ideas and experiments.

## Installation
1. clone this repository: `git clone https://github.com/minerva-ml/open-solution-avito-demand-prediction.git`
2. install requirements
3. register to [Neptune](https://neptune.ml/ 'machine learning lab') *(if you wish to use it)*
4. update `neptune.yaml` configuration file with your data filepaths
5. run experiment
*   with neptune:
```bash
$ neptune login
$ neptune experiment run --config neptune.yaml main.py -- train_evaluate_predict --pipeline_name main
```
collect submit from `/output/solution-1` directory.

* with pure python:
```bash
$ python main.py -- train_evaluate_predict --pipeline_name main
```

collect submit from `experiment_dir` directory that was specified in `neptune.yaml`

## Get involved
You are welcome to contribute your code and ideas to this open solution. To get started:
1. Check [competition project](https://github.com/minerva-ml/open-solution-avito-demand-prediction/projects/1) here, on GitHub to see what we are working on right now.
1. Express your interest in paticular task by writing comment in this task, or by creating new one with your fresh idea.
1. We will get back to you quickly in order to start working together.
1. Check [CONTRIBUTING](CONTRIBUTING.md) for some more information.

## User support
There are several ways to seek help:
1. Kaggle [discussion](https://www.kaggle.com/c/avito-demand-prediction/discussion) is our primary way of communication.
1. Read project's [Wiki](https://github.com/minerva-ml/open-solution-avito-demand-prediction/wiki), where we publish descriptions about the code, pipelines and supporting tools such as [neptune.ml](https://neptune.ml/).
1. Submit an [issue](https://github.com/minerva-ml/open-solution-avito-demand-prediction/issues) directly in this repo.
