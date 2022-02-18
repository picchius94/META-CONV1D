# =============================================================================
# PROJECT CHRONO - http://projectchrono.org
#
# Copyright (c) 2014 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http://projectchrono.org/license-chrono.txt.
#
# =============================================================================
# Authors: Radu Serban
# =============================================================================
#
# Demonstration of a steering path-follower and cruise control PID controlers in:
# 1) Deformable Terrain with Soil Contact Model (SCM)
# 2) Unstructured Geometry from Height Map
#
# The vehicle reference frame has Z up, X towards the front of the vehicle, and
# Y pointing to the left.
#
# =============================================================================

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as irr
import math as m
import numpy as np
import pandas as pd
import os
import terrain_list as tl
import random

# The path to the Chrono data directory containing various assets (meshes, textures, data files)
# is automatically set, relative to the default location of this demo.
# If running from a different directory, you must change the path to the data directory with: 
chrono.SetChronoDataPath("/home/server01/anaconda3/envs/marco_chrono/share/chrono/data/")
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')

# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = veh.VisualizationType_PRIMITIVES
suspension_vis_type =  veh.VisualizationType_PRIMITIVES
steering_vis_type = veh.VisualizationType_PRIMITIVES
wheel_vis_type = veh.VisualizationType_MESH
tire_vis_type = veh.VisualizationType_MESH 

# Type of powertrain model (SHAFTS, SIMPLE)
powertrain_model = veh.PowertrainModelType_SHAFTS

# Drive type (FWD, RWD, or AWD)
drive_type = veh.DrivelineTypeWV_AWD

# Steering type (PITMAN_ARM or PITMAN_ARM_SHAFTS)
steering_type = veh.SteeringTypeWV_PITMAN_ARM

# Type of tire model (RIGID, RIGID_MESH, PACEJKA, LUGRE, FIALA, PAC89)
tire_model = veh.TireModelType_RIGID

# Contact method
contact_method = chrono.ChContactMethod_NSC

def Q_to_E321(q):
    roll = m.atan2(2 * (q.e0 * q.e1 + q.e2 * q.e3), 1 - 2 * (q.e1**2 + q.e2**2))
    pitch = m.asin(2 * (q.e0 * q.e2 - q.e3 * q.e1))
    yaw = m.atan2(2 * (q.e0 * q.e3 + q.e1 * q.e2), 1 - 2 * (q.e2**2 + q.e3**2));
    return chrono.ChVectorD(roll, pitch, yaw)*180/m.pi;

# Map is divided in n terrain_types of equal length along x axis
class MySoilParams (veh.SoilParametersCallback):
    def __init__(self, terrain_types, Z_obst, map_size_x, map_size_y, discr, terrain_params_noise = 0):
        veh.SoilParametersCallback.__init__(self)
        self.discr = discr
        self.DEM_size_x = int(map_size_x/self.discr +1)
        self.DEM_size_y = int(map_size_y/self.discr +1)
        self.list_sp = []
        for terrain_type in terrain_types:
            sp = tl.Soil_Params(terrain_type)
            if sp is None:
                print("Invalid terrain type")
                return -1
            else:
                sp = np.array(sp)
                if terrain_params_noise:
                    for i in range(len(sp)):
                        sp[i] = sp[i] + sp[i]*terrain_params_noise*np.random.uniform(-1,1) 
                self.list_sp.append(sp)
        self.n_terrains = len(terrain_types)
        self.terrain_types = terrain_types
        self.Z_obst = Z_obst
    def Set(self, x, y):
        tp = (self.Z_obst[(np.floor((x-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                          (np.floor((y-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]).astype(int)
        sp = self.list_sp[tp]
        self.m_Bekker_Kphi = sp[0]
        self.m_Bekker_Kc = sp[1]
        self.m_Bekker_n = sp[2]
        self.m_Mohr_cohesion = sp[3]
        self.m_Mohr_friction = sp[4]
        self.m_Janosi_shear = sp[5]
        self.m_elastic_K = sp[6]
        self.m_damping_R = sp[7]    
        
    
            
#// =============================================================================

class simulator:
    def __init__(self, map_name, Z_obst, map_dims, init_loc, init_rot, terrain_type, terrain_params_noise = 0, visualisation = True):
        np.random.seed(0)
        # Simulation step sizes
        self.step_size = 2e-3
        self.tire_step_size = 1e-3
        
        # Other time variables
        self.delay = 0.5
        self.time_hand_brake = 0.5
        # Frequency data acquisition (in Hz)
        self.freq_data = 500
        
        # Other variables
        self.transient = True
        self.time = 0
        self.visualisation = visualisation
        
        # Output directories for state info
        self.state_output = False
        self.out_dir = "./STATISTICS/"
        self.out_file = "stats.csv"
        if self.state_output and not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        # Data Monitoring Vairiables
        self.data_list = []
        self.data = pd.DataFrame()
        self.data_steps = np.ceil(1/self.freq_data/self.step_size)
        
        # Default directory for maps
        self.default_terrain_dir = "./Terrains/"
        # Map settings and initial position
        if '.bmp' not in map_name:
            map_name += '.bmp'
        if os.path.exists(map_name):
            path_map = map_name
        elif os.path.exists(self.default_terrain_dir+map_name):
            path_map = self.default_terrain_dir+map_name
        else:
            print("Map Image not found!")
            return -1
        # Maps and initial pose initialisations
        map_size_x,map_size_y,map_height = map_dims
        x0, y0, z0 = init_loc
        roll0, pitch0, yaw0 = init_rot
        init_rot_quat = chrono.Q_from_AngZ(yaw0*m.pi/180)*chrono.Q_from_AngY(pitch0*m.pi/180)*chrono.Q_from_AngX(roll0*m.pi/180)
        
        #  Create the HMMWV vehicle, set parameters, and initialize
        self.my_hmmwv = veh.HMMWV_Full()
        self.my_hmmwv.SetContactMethod(contact_method)
        self.my_hmmwv.SetChassisFixed(False) 
        self.my_hmmwv.SetInitPosition(chrono.ChCoordsysD(chrono.ChVectorD(x0,y0,z0), init_rot_quat))
        self.my_hmmwv.SetPowertrainType(powertrain_model)
        self.my_hmmwv.SetDriveType(drive_type)
        self.my_hmmwv.SetSteeringType(steering_type)
        self.my_hmmwv.SetTireType(tire_model)
        self.my_hmmwv.SetTireStepSize(self.tire_step_size)
        self.my_hmmwv.Initialize()
        self.my_hmmwv.GetSystem().SetStep(self.step_size)
        self.my_hmmwv.GetSystem().SetNumThreads(1)
        if self.visualisation:
            self.my_hmmwv.SetChassisVisualizationType(chassis_vis_type)
            self.my_hmmwv.SetSuspensionVisualizationType(suspension_vis_type)
            self.my_hmmwv.SetSteeringVisualizationType(steering_vis_type)
            self.my_hmmwv.SetWheelVisualizationType(wheel_vis_type)
            self.my_hmmwv.SetTireVisualizationType(tire_vis_type)
        # Add mass to compensate for unbalance on right-side
        body_b = chrono.ChBody()
        mass = 400
        body_b.SetMass(mass)
        body_b.SetInertiaXX(chrono.ChVectorD(0.2, 0.2, 0.2)*(mass/6))
        body_b.SetPos(chrono.ChFrameD(chrono.ChVectorD(x0,y0,z0),init_rot_quat)*chrono.ChVectorD(0, -0.55, 1.55))
        body_b.SetRot(init_rot_quat)
        self.my_hmmwv.GetSystem().Add(body_b)
        mylink = chrono.ChLinkMateFix()
        mylink.Initialize(body_b, self.my_hmmwv.GetChassisBody())
        self.my_hmmwv.GetSystem().Add(mylink)
        
        
        # Create the terrain
        self.terrain_params_noise = terrain_params_noise
        self.terrain = veh.SCMDeformableTerrain(self.my_hmmwv.GetSystem())
        self.terrain.Initialize(path_map, map_size_x, map_size_y, 0.0, map_height, 0.05)
        if not hasattr(terrain_type, "__iter__"):
            terrain_type = [terrain_type, 9]
        else:
            terrain_type = terrain_type.extend(9)
        
        self.my_params = MySoilParams(terrain_type, Z_obst, map_size_x, map_size_y, 0.0625, terrain_params_noise)
        self.terrain.RegisterSoilParametersCallback(self.my_params)
        if visualisation:
            self.terrain.SetPlotType(veh.SCMDeformableTerrain.PLOT_SINKAGE, 0, 0.15)
            self.terrain.SetTexture(path_map[:-4]+'_obst.png')
        
        # My speed and steering controller gains
        self.Kp = 3
        self.Ki = 0.25
        self.Kd = 0.1
        self.integral_prior = 0
        self.error_prior = 0
        self.LookAheadDistance = 0.5
        self.last_steering = 0
        self.max_freq_steering = 0.05 # max steering variation e.g Dsteering per simulation step size
        
        # Visualisation application
        if visualisation:
            # Create the vehicle Irrlicht interface
            self.app = veh.ChWheeledVehicleIrrApp(self.my_hmmwv.GetVehicle(), 'HMMWV', irr.dimension2du(1000,800))
            #self.app = veh.ChWheeledVehicleIrrApp(self.my_hmmwv.GetVehicle(), 'HMMWV', irr.dimension2du(1600,900))
            self.app.SetSkyBox()
            #self.app.SetHUDLocation(0,0)
            #self.app.EnableStats(False)
            self.app.AddTypicalLights(irr.vector3df(-60, -30, 100), irr.vector3df(60, 30, 100), 250, 130)
            self.app.AddTypicalLogo(chrono.GetChronoDataFile('logo_pychrono_alpha.png'))
            self.app.SetChaseCamera(chrono.ChVectorD(0.0, 0.0, 1.75), 6.0, 0.5)
            self.app.SetTimestep(self.step_size)
            self.app.AssetBindAll()
            self.app.AssetUpdateAll()
        
           	# Visualization of controller points (sentinel & target)
            self.ballS = self.app.GetSceneManager().addSphereSceneNode(0.1);
            self.ballT = self.app.GetSceneManager().addSphereSceneNode(0.1);
            self.ballS.getMaterial(0).EmissiveColor = irr.SColor(0, 255, 0, 0);
            self.ballT.getMaterial(0).EmissiveColor = irr.SColor(0, 0, 255, 0);
        
        
    def set_state_output(self, state_output, out_dir = None, out_file = None):
        self.state_output = state_output
        if out_dir is not None: self.out_dir = out_dir
        if out_file is not None: self.out_file = out_file
        if self.state_output and not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)
        
    def close(self):
        if self.visualisation:
            self.app.GetDevice().closeDevice()
            if not self.app.GetDevice().run():
                pass
            
    def run(self, path_points, target_speed):
        x_v, y_v, z_v = path_points
        start_point = chrono.ChVectorD(x_v[0],y_v[0],z_v[0])
        end_point = chrono.ChVectorD(x_v[-1],y_v[-1],z_v[-1])
        trajectory = chrono.vector_ChVectorD()
        for xx, yy, zz in zip(x_v,y_v,z_v):
            trajectory.push_back(chrono.ChVectorD(xx,yy,zz))
        trajectory_B = chrono.ChBezierCurve(trajectory)
        
        # Create the path-follower, cruise-control driver    
        driver = veh.ChPathFollowerDriver(self.my_hmmwv.GetVehicle(), trajectory_B, "my_path", target_speed)
        driver.GetSteeringController().SetLookAheadDistance(self.LookAheadDistance)
        driver.GetSteeringController().SetGains(5, 0, 0)
        driver.GetSpeedController().SetGains(0.4, 0, 0) # not used
        driver.Initialize()
        
    
        if self.visualisation:
            self.app.AssetBindAll()
            self.app.AssetUpdateAll()
        
        # Initialize simulation frame counter and simulation time
        end = False
        goal_reached = False
        sim_frame = 0
        min_distance = 100000
        count_distance_tolerance = 0
        t_end = self.time + self.delay + 1 + 1.5*m.sqrt((start_point.x - end_point.x)**2 + (start_point.y - end_point.y)**2)/target_speed;
        self.data_list_run = []
        self.data_run = pd.DataFrame()
        # Simulation loop
        #realtime_timer = chrono.ChRealtimeStepTimer()
        while (True):
            if self.visualisation:
                if not self.app.GetDevice().run():
                    break
            time = self.my_hmmwv.GetSystem().GetChTime()
            self.time = time
            speed = self.my_hmmwv.GetVehicle().GetChassisBody().GetPos_dt()
            q = self.my_hmmwv.GetVehicle().GetChassisBody().GetRot()
            speed_rel = q.GetInverse().Rotate(speed)
            eul321 = Q_to_E321(q)
            
            motor_speed = self.my_hmmwv.GetPowertrain().GetMotorSpeed();
            motor_torque = self.my_hmmwv.GetPowertrain().GetMotorTorque();
            # Update sentinel and target location markers for the path-follower controller.
            # Note that we do this whether or not we are currently using the path-follower driver.
            pS = driver.GetSteeringController().GetSentinelLocation()
            pT = driver.GetSteeringController().GetTargetLocation()
            lat_distance = m.sqrt((pS.x - pT.x)**2 + (pS.y - pT.y)**2)
            if (sim_frame == 0):
                lat_distance = 0
                
            if time < self.delay: # initial time while doing nothing
                self.my_hmmwv.GetVehicle().ApplyParkingBrake(True)
                driver.SetThrottle(0)
                driver.SetBraking(0)
            else:
                # Implementing PID for speed controller
                error = target_speed - speed_rel.x
                integral = self.integral_prior + error * self.step_size
                derivative = (error - self.error_prior) / self.step_size
                self.error_prior = error
                self.integral_prior = integral
                command = self.Kp * error + self.Ki * integral + self.Kd * derivative
                if command < 0:
                    driver.SetBraking(-command)
                    driver.SetThrottle(0)
                else:
                    driver.SetBraking(0)
                    driver.SetThrottle(command)
                # Starting with hand brake on is useful on inclined terrains
                if time < self.time_hand_brake and eul321.y < -7:
                    self.my_hmmwv.GetVehicle().ApplyParkingBrake(True)
                elif self.transient:
                    self.my_hmmwv.GetVehicle().ApplyParkingBrake(False)
                    self.transient = False
                
            # Conditions for simulation end
            pos = self.my_hmmwv.GetVehicle().GetVehiclePos()
            distance = m.sqrt((pos.x - end_point.x)**2 + (pos.y - end_point.y)**2)
            if distance < self.LookAheadDistance:
                driver.SetSteering(self.last_steering) #tweak to handle steering control when target has reached the end
                # I try to get closer and closer
                if distance <= min_distance:
                    min_distance = distance
                    count_distance_tolerance = 0
                else:
                    # Stop if distance rises for more than 10 consecutive frames
                    count_distance_tolerance += 1
                    if count_distance_tolerance > 10:
                        goal_reached = True
                        end = True
            elif min_distance < self.LookAheadDistance:
                # This can happen if I reached the target but, by trying to get closer, I get out of it
                goal_reached = True
                end = True
            else:
                # This handles cases when robot misses the target
                end_direction = m.atan2(end_point.y - pos.y, end_point.x - pos.x) * 180/m.pi - eul321.z
                if end_direction > 180:
                    end_direction -= 360
                elif end_direction < -180:
                    end_direction += 360
                if abs(end_direction) > 90:
                    end = True
            if time >= t_end or (time > self.delay+1 and speed_rel.x < 0.1):
                # This handles cases when robot gets stuck
                if distance < self.LookAheadDistance:
                    goal_reached = True
                end = True
                
            # tweak to handle steering control when transitioning from one action to another
            if sim_frame < 5:
                driver.SetSteering(self.last_steering)
            # tweak to handle steering control when transitioning from one action to another
            current_steering = driver.GetSteering()
            if abs(current_steering-self.last_steering) > self.max_freq_steering:
                if current_steering > self.last_steering:
                    current_steering = self.last_steering+self.max_freq_steering
                    driver.SetSteering(current_steering)
                else:
                    current_steering = self.last_steering-self.max_freq_steering
                    driver.SetSteering(current_steering)
            self.last_steering = current_steering
                
            #self.last_steering = driver.GetSteering()
            
            if sim_frame % self.data_steps == 0:
                self.data_list.append({"Time": time, "I_Steering": driver.GetSteering(),
                                    "I_Throttle": driver.GetThrottle(), "I_Braking": driver.GetBraking(),
                                    "FWD_Speed": speed_rel.x, "Lat_Speed": speed_rel.y,
                                    "Lat_Distance": lat_distance,
                                    "X": pos.x, "Y": pos.y, "Yaw": eul321.z, "Roll": eul321.x, "Pitch": eul321.y,
                                    "Motor_Speed": motor_speed, "Motor_Torque": motor_torque})
                self.data_list_run.append({"Time": time, "I_Steering": driver.GetSteering(),
                                    "I_Throttle": driver.GetThrottle(), "I_Braking": driver.GetBraking(),
                                    "FWD_Speed": speed_rel.x, "Lat_Speed": speed_rel.y,
                                    "Lat_Distance": lat_distance,
                                    "X": pos.x, "Y": pos.y, "Yaw": eul321.z, "Roll": eul321.x, "Pitch": eul321.y,
                                    "Motor_Speed": motor_speed, "Motor_Torque": motor_torque})
            
            if self.visualisation:
                # Draw scene
                self.ballS.setPosition(irr.vector3df(pS.x, pS.y, pS.z))
                self.ballT.setPosition(irr.vector3df(pT.x, pT.y, pT.z))
                self.app.BeginScene(True, True, irr.SColor(255, 140, 161, 192))
                self.app.DrawAll()
                self.app.EndScene()
    
            # Update modules (process inputs from other modules)
            driver_inputs = driver.GetInputs()
            driver.Synchronize(time)
            self.terrain.Synchronize(time)
            self.my_hmmwv.Synchronize(time, driver_inputs, self.terrain)
            if self.visualisation:
                self.app.Synchronize("", driver_inputs)
    
            # Advance simulation for one timestep for all modules
            driver.Advance(self.step_size)
            self.terrain.Advance(self.step_size)
            self.my_hmmwv.Advance(self.step_size)
            if self.visualisation:
                self.app.Advance(self.step_size)
    
            # Spin in place for real time to catch up
            #realtime_timer.Spin(step_size)
            sim_frame += 1
        
            if end:
                break
        
        if self.visualisation:
            driver.GetSteeringController().Reset(self.my_hmmwv.GetVehicle())
            self.app.AssetBindAll()
            self.app.AssetUpdateAll()
        
        self.data = pd.DataFrame(self.data_list)
        self.data_run = pd.DataFrame(self.data_list_run)
        if self.state_output:
           self.data.to_csv(self.out_dir+self.out_file, index=False)
           
           
        return goal_reached


