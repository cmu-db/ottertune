# OtterTune

[![Build Status](https://travis-ci.org/cmu-db/ottertune.svg?branch=master)](https://travis-ci.org/cmu-db/ottertune)
[![codecov.io](https://codecov.io/github/cmu-db/ottertune/coverage.svg?branch=master)](https://codecov.io/github/cmu-db/ottertune)

OtterTune is a new tool developed by students and researchers in the [Carnegie Mellon Database Group](http://db.cs.cmu.edu/projects/autotune/) that can automatically find good settings for a database management system's configuration knobs. The goal is to make it easier for anyone to deploy a DBMS without any expertise in database administration. To tune new DBMS deployments, OtterTune reuses training data gathered from previous tuning sessions. Because OtterTune does not need to generate an initial dataset for training its ML models, tuning time is drastically reduced.

For more information, see our [paper](http://db.cs.cmu.edu/papers/2017/p1009-van-aken.pdf).

```
@inproceedings{vanaken17,
  author = {Van Aken, Dana and Pavlo, Andrew and Gordon, Geoffrey J. and Zhang, Bohan},
  title = {Automatic Database Management System Tuning Through Large-scale Machine Learning},
  booktitle = {Proceedings of the 2017 ACM International Conference on Management of Data},
  series = {SIGMOD '17},
  year = {2017},
  pages = {1009--1024},
  numpages = {16},
 }
 ```

## Contributors

See the [people page](https://github.com/cmu-db/ottertune/graphs/contributors) for the full list of contributors.
