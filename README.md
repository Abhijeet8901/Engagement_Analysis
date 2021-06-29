# Engagement Analysis
This is a private repository for I'mbesideyou project - **Engagement Analysis**.

## Requirements
* mxnet

The file ***main_code.py*** cotains all the functions needed.

The Driver function is in the file **Driver Function.ipnyb**.

The Driver Function takes input one necessary argument - Path to Video File , and one optional argument - max_eye_offset(Default Value = 25. The more it's value the more eye can look away from screen without getting labeled Disangeged.)

It outputs a list with length - Number of frames in input Video.

Each element of the list is a list of three elements containing these 3 info about that frame - (Label, Start Time of frame, End Time of frame)

* Label - A String valued "Engaged" or "DisEngaged"
* Start Time of frame - Integer containing start time of frame realtive to start of video.
* End Time of frame - Integer containing end time of frame realtive to start of video.

In the Driver Function, change the Variable name ***path_to_Engagement_Model*** to the path of Engagement Model Directory
