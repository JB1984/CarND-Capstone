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

# Architecuture
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
1. In Udacity workspace simulator, there is always obvious lagging when camera is on. This seems to be due to workspace resource constraint. The solution is to set up local simulator enviroment.  Several members in our team have done this successfully.
