#!/usr/bin/env bash

########################
#  Build docker
#######################

mkdir src/regroupement
cp -r ../../regroupement/training  src/regroupement
cp -r ../../regroupement/inference  src/regroupement
cp -r ../../regroupement/data_preparation  src/regroupement
cp -r ../../regroupement/utils.py  src/regroupement
mkdir src/prediction_models
cp -r ../../prediction_models/inference src/prediction_models
cp ../../prediction_models/config.yaml src/prediction_models

#echo "Building CPU version"
docker build -t starclay/dgs_backend:latest .

rm -rf  src/prediction_models
rm -rf  src/regroupement