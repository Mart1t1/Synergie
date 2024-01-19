# AIOnIce

A figure skating jump recognition software using IMU data as input. 

CLI usage: 

```sh
pip install requirements.txt
python3 mai.py -p <session name> | -t
```

-p is used for the dataset generation.
-t is used for training.

## API

A fastAPI is available inside the api/api.py file. It only has one endpoint.


## How to add data

Add a new dataset in constants.sessions.
'path' is the path were all csvs of the session are. IMU acquisition has to be synchronized.

'sample_time_fine_synchro' corresponds to the sample_time_fine synchro where video begins (so your annotation ready data is in sync with the video).


## About the dataset

Current model has been trained with a dataset of roughly 500 annodated jumps.
Because of privacy concerns, the dataset is not disclosed.


## Credits

Made by the S2M for Patinage Quebec.
