# Simple Online Realtime Tracker (SORT)
SORT is an implementation of Kalman Filter based realtime object tracker. It uses dynamic model of the environment ( position, velocity, acceleration ) to estimate the future positions of object, then it uses Kalman Filters to correct its estimates.

## Dependencies

```
python3
numpy
scipy
opencv
matplotlib
tqdm
pytorch
torchvision
```

## Run the tracker
- Clone repository
```
git clone https://github.com/kushwahashivam/SORT.git
```
- Run the run.py file
```
cd SORT
python run.py
```

## Change the video source
- Open config.py file.
- assign camera number (0, 1 etc) to 'source' variable.
- OR, to use video as source, video path to 'source' variable.