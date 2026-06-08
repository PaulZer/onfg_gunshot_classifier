# [WIP] ONG Gunshot classifier

Attempt to use [tropical_forest_gunshot_classifier](https://github.com/lydiakatsis/tropical_forest_gunshot_classifier), a model trained to automate detection of gunshots in tropical forests, using convolutional neural networks.

See the original repository for implementation details, dataset references, and licensing/attribution information.

## Goals
 Better conservation and hunting monitoring in French Guiana for ONF Guyane. 

# Usage

## 1. Docker setup
Build docker image and launch compose project (`docker_compose.yml`) 

## 2. Prepare audio to analyze
Drop your audio files in the `audio_to_analyze` folder.

## 3. Execute classify algorithm
Once the docker container is up and runnning, run the script `classify_gunshots`
This will output a `.csv` file, referencing gunshot detection events, for each file dropped in the `audio_to_analyze` folder, with the confidence score on each detection event.

# Resources

- [tropical_forest_gunshot_classifier Github repository](https://github.com/lydiakatsis/tropical_forest_gunshot_classifier)
- [Original classifying script on Google Colab](https://colab.research.google.com/github/lydiakatsis/tropical_forest_gunshot_classifier/blob/main/Gunshot%20classification%20GUI/Gunshot_classifier_colab.ipynb)

