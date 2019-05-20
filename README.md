# Programming a Real Self-Driving Car
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
# Team Aurora
- Ying LI	: liyingcmu@gmail.com
- Christian Welling	: csw73@cornell.edu			
- Jason Benner:	jbenner1984@gmail.com
- Erik Uckert:	neffe.donald@gmail.com
- Peng WU	: wupeng1510@163.com


# Description
This project is to implement a self-driving car system with ROS to run in both simulator and Udacity’s self-driving car - Carla.

# Architecture
Implemented ROS nodes for core functionality of the autonomous vehicle system:
- waypoint following,
- control,
- traffic light detection

# ROS Nodes
- waypoint_updater: This node subscribes to the /base_waypoints, /current_pose, /obstacle_waypoint, and /traffic_waypoint topics, and publish a list of waypoints ahead of the car with target velocities to the /final_waypoints topic
- tl_detector: This node takes in data from the /image_color, /current_pose, and /base_waypoints topics and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic
- dbw_node: subscribes to the /current_velocity topic along with the /twist_cmd topic to receive target linear and angular velocities. Additionally, this node subscribes to /vehicle/dbw_enabled, which indicates if the car is under dbw or driver control. This node publishes throttle, brake, and steering commands to the /vehicle/throttle_cmd, /vehicle/brake_cmd, and /vehicle/steering_cmd topics.


# How to launch
There are two tracks in simulator.
## highway track
```
roslaunch launch/styx.launch
```

## test site:
first modify waypoint_loader/launch/waypoint_loader.launch to :
```
<param name="path" value="$(find styx)../../../data/churchlot_with_cars.csv"/>
```
and then launch:
```
roslaunch launch/site.launch
```

# Learnings
## 1. Working Environments 
In the Udacity workspace simulator, our team has experienced obvious lagging when the camera is on. This seems to be due to workspace resource constraints. The solution was to set up a local simulator environment.  There are several ways to go. One can set up a native Linux installation with both, simulator and ROS environment, running on the same machine. Because our team had no option to set up a complete native Linux installation, our way to go was using a docker installation and also the provided virtual machine. 

## 1.1 General Tweaks
For us, it was necessary to change some timing relevant variables in the original codebase. First, the variable [``LOOKAHEAD_WPS``](https://github.com/yingCMU/CarND-Capstone/blob/fd09afc5be55fddcb9ffcc96eef047dfbf518b57/ros/src/waypoint_updater/waypoint_updater.py#L27) must be reduced below a value of 200. A Value of 100 was satisfactory for not running into lag issues while computing enough waypoints ahead. In addition, we have to change [``LOOP_RATE``](https://github.com/yingCMU/CarND-Capstone/blob/fd09afc5be55fddcb9ffcc96eef047dfbf518b57/ros/src/waypoint_follower/src/pure_pursuit.cpp#L33-L35) up to 50 Hz to increase the performance. In the original Udacity code the [``PurePursuit::calcTwist()``](https://github.com/yingCMU/CarND-Capstone/blob/fd09afc5be55fddcb9ffcc96eef047dfbf518b57/ros/src/waypoint_follower/src/pure_pursuit_core.cpp#L252-L276) function calculates new values in dependence of the [``PurePursuit::verifyFollowing()``](https://github.com/yingCMU/CarND-Capstone/blob/fd09afc5be55fddcb9ffcc96eef047dfbf518b57/ros/src/waypoint_follower/src/pure_pursuit_core.cpp#L232-L251) function. To avoid strange effects of "lane wandering" in combination with a [``LOOP_RATE``](https://github.com/yingCMU/CarND-Capstone/blob/fd09afc5be55fddcb9ffcc96eef047dfbf518b57/ros/src/waypoint_follower/src/pure_pursuit.cpp#L33-L35) value of 30 Hz we altered it to basically update at every step regardless of the lane following verification.
