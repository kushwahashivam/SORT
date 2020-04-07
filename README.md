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
- OR, to use video as source, assign video path to 'source' variable.

## Citing SORT

If you find this repo useful in your research, please consider citing:

    @inproceedings{Bewley2016_sort,
      author={Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben},
      booktitle={2016 IEEE International Conference on Image Processing (ICIP)},
      title={Simple online and realtime tracking},
      year={2016},
      pages={3464-3468},
      keywords={Benchmark testing;Complexity theory;Detectors;Kalman filters;Target tracking;Visualization;Computer Vision;Data Association;Detection;Multiple Object Tracking},
      doi={10.1109/ICIP.2016.7533003}
    }