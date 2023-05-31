# Overview #

The goal of those libraries is to enable easy integration with cloud services and deploy models, and aslo set of easy-to-use utilities.

## Installing airmelt_utils package ##

`pip install --ignore-installed git+ssh://git@github.com/semeonb/airmelt_utils.git`

## Example on how to import the packages ###

`from airmelt_data import gcp, azure, aws`
`from airmelt_system import toolbox, parallel`

# Libraries

## airmelt_data ##
Cloud services and ML models library.

### aws ###
Google Cloud functions

### gcp ###
Amazon Web Services functions

### azure ###
MS Azure functions

### db ###
Database functions

### dataprep ###
Data preparation functions

## models ###
Set of classes with various ML models available

### ml ###
A generic class that can use any model from the models package and create prediction / training

### existing ###
A packge that neables work with exisitng models


# Version Updates #

### Version 0.0.2 ###
Adding Slack module

### Version 0.1.0 ###
Adding comments and running through QA

### Version 0.1.1 ###
Adding comments and adding df2bq method

### Version 0.1.2 ###
Adding XGboost model

### Version 0.1.2 ###
Adding Dot Embed  model