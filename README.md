# GuidedFiltering
Python Implementation of Guided Image Filter

This code is implemented based on the Matlab reference code provided by Kaiming He et al.
For more details, please refer to the project [[Guided Image Filter]](http://kaiminghe.com/eccv10/index.html).

### Installation
*Note*: These codes were only tested on **Windows7** with **Python 3.5**.

### Dependencies
Please install Anaconda3 4.2.0.

### Usage
```python
python example_feathering.py
```

### Results
Some differences will appear, because of the **inv** method is different between MATLAB and Numpy.

### Acknowledge
Heavily borrowed from this project [[pfcai]](https://github.com/pfchai/GuidedFilter).
However, the above only implements **guidedfilter** and doesn't implement **guidedfilter_color**.
