import numpy as np
import math
import heapq
import time
import ray
import psutil
import pandas as pd
import models
from scipy import interpolate
from scipy.stats import gamma

from copy import copy


class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]
    
    def get_best_priority(self):
        if not self.empty():
            element = heapq.heappop(self.elements)
            heapq.heappush(self.elements, element)
            return element[0]
        else:
            return np.inf
    
class A_star:
    def __init__(self, params):
        # State discretization for A* OPEN que
        self.discr_th = math.pi/180
        self.discr_x = 0.01
        self.discr_y = 0.01
        # Robot dims
        self.wheelbase = 1.688965*2
        self.wheeltrack = 0.95*2
        self.eps_base = 0.5
        self.eps_track = 0.54
        self.extra_eps = 2
        self.belly = 0.5
        self.discr_voxel = 0.0625
        self.wheeltrace_cells = 20  # should be <= to: int((wheeltrack+eps_track)/discr_voxel +1)/2
        self.wheel_around_cells = 3
        # Action space
        self.num_actions = 5
        self.length = 2.7
        self.max_angle = 20*math.pi/180
        self.forward_actions = []
        if self.num_actions > 1:
            self.curvature_list = np.linspace(-self.max_angle/self.length,self.max_angle/self.length,self.num_actions)
        else:
            self.curvature_list = np.array([0])
        for curvature in self.curvature_list:
            if curvature:
                self.forward_actions.append((1/curvature,self.length*curvature*180/math.pi))
            else:
                self.forward_actions.append((self.length,0))
        # Sortying actions by increasing curvature
        self.actions = list(range(self.num_actions))
        self.actions_sorted = []
        while self.actions:
            id_min = abs(self.curvature_list[self.actions]).argmin()
            self.actions_sorted.append(self.actions.pop(id_min))
        # Goal setting
        self.goal_type = params["goal_type"]
        self.goal_radius = params["goal_radius"]
        self.goal_depth = params["goal_depth"]
        self.goal_width = params["goal_width"]
        # Each action is segmented with follwoing params
        self.segment_length = 0.9
        self.n_points = 49
        self.n_segments = int(self.length/self.segment_length)
        self.segment_points = int((self.n_points-1)//self.n_segments)
        
        # Scaling of meta heuristic to enforce underestimate
        self.meta_h_scaling = params["meta_h_scaling"]
        # Available optimization criteria
        self.all_opt_criterias = ["energy", "pitch", "distance"]
        
        self.params = params
            
        try:
            ray.init(num_cpus=psutil.cpu_count(logical=True))
        except:
            ray.shutdown()
            ray.init(num_cpus=psutil.cpu_count(logical=True))
            
        self.log_time_period = 1
        self.timeout = 60
        
        # Memory shots
        self.all_methods = []
        self.params_model = {}
        self.model = {}
        self.memory_list = {}
        self.memory = {}
        self.memory_geom_merged_list = {}
        # self.memory_geom_list = {}
        for method in params["WHICH_METHOD"]:
            self.memory_list[method] = []
            self.memory[method] = pd.DataFrame()
            self.memory_geom_merged_list[method] = []
            # self.memory_geom_list[method] = []
        
        # Robot Base Settings
        self.mask_map_left = np.arange(self.extra_eps//2,self.wheeltrace_cells+self.extra_eps//2)
        self.mask_map_right = np.arange(-self.wheeltrace_cells-self.extra_eps//2,-self.extra_eps//2)
        self.mask_map = np.concatenate([self.mask_map_left,self.mask_map_right])
        self.BASE_size = int((self.wheelbase+self.eps_base)/self.discr_voxel + 1)
        self.TRACK_size = int((self.wheeltrack+self.eps_track)/self.discr_voxel + 1)
        self.Z_map_size = (self.BASE_size+self.extra_eps, self.TRACK_size+self.extra_eps)
        self.X_map, self.Y_map = np.meshgrid(np.arange(0, self.Z_map_size[1]), np.arange(0, self.Z_map_size[0]))
        self.wheel_base_tot = self.wheelbase+self.eps_base+self.extra_eps*self.discr_voxel
        self.wheel_track_tot = self.wheeltrack+self.eps_track+self.extra_eps*self.discr_voxel
        self.x_local = np.linspace(-self.wheel_base_tot/2,self.wheel_base_tot/2,self.BASE_size+self.extra_eps)
        self.y_local = np.linspace(-self.wheel_track_tot/2,self.wheel_track_tot/2,self.TRACK_size+self.extra_eps)
        self.Y_local, self.X_local = np.meshgrid(self.y_local,self.x_local)
        self.l1 = self.BASE_size*self.discr_voxel/2
        self.l2 = self.l1+self.length
        self.W_trace_size = (self.BASE_size+self.n_segments*self.segment_points-1,self.wheeltrace_cells*2)
        self.x_wheels_loc = np.array([self.wheelbase/2, self.wheelbase/2, -self.wheelbase/2, -self.wheelbase/2])
        self.y_wheels_loc = np.array([-self.wheeltrack/2, self.wheeltrack/2, -self.wheeltrack/2, self.wheeltrack/2])
        ix_f = np.floor((self.wheelbase/2-(-self.Z_map_size[0]*self.discr_voxel/2))/self.discr_voxel).astype(int)
        ix_r = np.floor((-self.wheelbase/2-(-self.Z_map_size[0]*self.discr_voxel/2))/self.discr_voxel).astype(int)
        iy_r = np.floor((self.wheeltrack/2-(-self.Z_map_size[1]*self.discr_voxel/2))/self.discr_voxel).astype(int)
        iy_l = np.floor((-self.wheeltrack/2-(-self.Z_map_size[1]*self.discr_voxel/2))/self.discr_voxel).astype(int)
        self.x_around_f = self.x_local[np.arange(-self.wheel_around_cells+ix_f,self.wheel_around_cells+ix_f+1)]
        self.x_around_r = self.x_local[np.arange(-self.wheel_around_cells+ix_r,self.wheel_around_cells+ix_r+1)]
        self.y_around_l = self.y_local[np.arange(-self.wheel_around_cells+iy_l,self.wheel_around_cells+iy_l+1)]
        self.y_around_r = self.y_local[np.arange(-self.wheel_around_cells+iy_r,self.wheel_around_cells+iy_r+1)]
        self.y_around_fl, self.x_around_fl = np.meshgrid(self.y_around_l,self.x_around_f)
        self.y_around_fr, self.x_around_fr = np.meshgrid(self.y_around_r,self.x_around_f)
        self.y_around_rl, self.x_around_rl = np.meshgrid(self.y_around_l,self.x_around_r)
        self.y_around_rr, self.x_around_rr = np.meshgrid(self.y_around_r,self.x_around_r)
        self.x_around_wheels = np.concatenate([self.x_around_fl.flatten(),self.x_around_fr.flatten(),self.x_around_rl.flatten(),self.x_around_rr.flatten()])
        self.y_around_wheels = np.concatenate([self.y_around_fl.flatten(),self.y_around_fr.flatten(),self.y_around_rl.flatten(),self.y_around_rr.flatten()])
        
            
    
    def set_models(self, method, params_model):
        import tensorflow as tf
        flag_conv1D = False
        flag_pr = False
        
        self.all_methods.append(method)
        self.params_model[method] = params_model
        self.method = method
        if not "HEURISTIC_TYPE" in self.params_model[self.method]:
            self.params_model[self.method]["HEURISTIC_TYPE"] = "opt"
        
        # This is done only to initialize ray (so that the first actual time will be much faster)
        if 'conv1D' in method and not flag_conv1D:
            flag_conv1D = True
            ray.get([self.merged_wheel_trace.remote(self, (0,0,0), 0) for x in range(self.num_actions)])
        elif 'plane' in method and not flag_pr:
            flag_pr = True
            ray.get([self.merged_pitch_roll.remote(self, (0,0,0), 0) for x in range(self.num_actions)])
        
        if 'meta' in method:
            self.model[method] = models.get_model(params_model, summary=True)
            self.model[method].load_weights(params_model["model_weights"])
            
            # Following is done only for initializing model (so that the first actual use will be faster)
            input_shape = [inp.shape.as_list() for inp in self.model[method].input]
            dummy_input = []
            for i in range(len(input_shape)):
                input_shape[i][0] = self.num_actions #dummy batch size
                dummy_input.append(tf.zeros(input_shape[i]))
            self.model[method](dummy_input)
        elif 'ST' in method:
            self.model[method] = []
            flag_summary = True
            for sep_model in params_model["sep_models"]:
                self.model[method].append(models.get_model(params_model, summary=flag_summary))
                flag_summary = False
                self.model[method][-1].load_weights(sep_model["model_weights"])
                
                # Following is done only for initializing model (so that the first actual use will be faster)
                input_shape = self.model[method][-1].input.shape.as_list()
                input_shape[0] = self.num_actions #dummy batch size
                dummy_input = tf.zeros(input_shape)
                self.model[method][-1](dummy_input)
            
            
        
    def set_map(self, Z, map_size_x, map_size_y, discr):
        # Map setting
        self.Z = Z
        self.map_size_x = map_size_x
        self.map_size_y = map_size_y
        self.discr = discr
        self.DEM_size_x = int(self.map_size_x/self.discr +1)
        self.DEM_size_y = int(self.map_size_y/self.discr +1)
        self.x = np.linspace(-self.map_size_x/2,self.map_size_x/2,num=self.DEM_size_x)
        self.y = np.linspace(-self.map_size_y/2,self.map_size_y/2,num=self.DEM_size_y)
        self.Y , self.X = np.meshgrid(self.y,self.x)
        
    def add_memory_shot(self, shot, method = None):
        if method is None:
            method = self.method
        # self.memory_geom_list[method].append(shot.pop('wheel_trace'))
        if 'wheel_trace_tot' in shot:
            self.memory_geom_merged_list[method].append(shot.pop('wheel_trace_tot'))
        self.memory_list[method].append(shot)
        
    def update_memory(self, method = None):
        if method is None:
            method = self.method
            
        if self.memory_list[method]:
            self.memory[method] = pd.DataFrame(self.memory_list[method])
    
    def norm_angle(self,theta):
        if theta > math.pi:
            theta -= 2*math.pi
        elif theta < -math.pi:
            theta += 2*math.pi
        return theta
    
    def action2traj(self,id_a,x0,y0,yaw0, n_points= 49):
        (r, dyaw) = self.forward_actions[id_a]
        dyaw = dyaw*math.pi/180
        yaw = np.linspace(yaw0,yaw0+dyaw,n_points)
        yaw = np.array(list(map(self.norm_angle, yaw)))
        if dyaw:
            x = x0 - r*np.sin(yaw0) + r*np.sin(yaw);
            y = y0 + r*np.cos(yaw0) - r*np.cos(yaw);
        else:
            x = np.linspace(x0, x0 + r*np.cos(yaw0), n_points);
            y = np.linspace(y0, y0 + r*np.sin(yaw0), n_points);
        return x,y,yaw
    
    def all_wheels(self, x0, y0, yaw0):
        x_wheels_global = self.x_wheels_loc*np.cos(yaw0) - self.y_wheels_loc*np.sin(yaw0) + x0
        y_wheels_global = self.x_wheels_loc*np.sin(yaw0) + self.y_wheels_loc*np.cos(yaw0) + y0
        
        #front_left, front_right, rear_left, rear_right
        return [[xw,yw] for xw,yw in zip(x_wheels_global,y_wheels_global)]
        

    def points(self, start, id_a, n_points = 49):
        x_ref = []
        y_ref = []
        yaw_ref = []
        xi, yi, yawi = start
        for ida in id_a:
            xi_ref, yi_ref, yawi_ref = self.action2traj(ida, xi, yi, yawi, n_points)
            if x_ref:
                xi_ref = xi_ref[1:]
                yi_ref = yi_ref[1:]
                yawi_ref = yawi_ref[1:]
            x_ref.extend(list(xi_ref))
            y_ref.extend(list(yi_ref))
            yaw_ref.extend(list(yawi_ref))
            xi = x_ref[-1]
            yi = y_ref[-1]
            yawi = yaw_ref[-1]
        return x_ref, y_ref, yaw_ref
    
    def normalize_input_features(self, X):
        for i in range(X.shape[-1]):
            X[...,i] = (X[...,i]-self.params_model[self.method]["INPUT_FEATURES_val1"][i])/self.params_model[self.method]["INPUT_FEATURES_val2"][i]
        return X
    def normalize_input_features_separate(self, X, sep_model):
        model = self.params_model[self.method]["sep_models"][sep_model]
        for i in range(X.shape[-1]):
            X[...,i] = (X[...,i]-model["INPUT_FEATURES_val1"][i])/model["INPUT_FEATURES_val2"][i]
        return X
        
    def normalize_extra_info(self, X):
        for i in range(X.shape[-1]):
            X[...,i] = (X[...,i]-self.params_model[self.method]["SHOTS_EXTRA_INFO_val1"][i])/self.params_model[self.method]["SHOTS_EXTRA_INFO_val2"][i]
        return X
        
    def normalize_energy(self, energy):
        try:
            energy += self.params_model[self.method]["EPS"]
        except:
            pass
        energy = (energy-self.params_model[self.method]["energy_val1"])/self.params_model[self.method]["energy_val2"]
        if 'shift' in self.params_model[self.method]["ENERGY_NORM_TYPE"]:
            energy += self.params_model[self.method]["energy_val1"]/self.params_model[self.method]["energy_val2"]
        return energy
   
    def denormalize_energy_tot(self, energy):
        if 'shift' in self.params_model[self.method]["ENERGY_NORM_TYPE"]:
            energy -= self.params_model[self.method]["energy_val1"]/self.params_model[self.method]["energy_val2"]*self.params_model[self.method]["LENGTH_SEQUENCE"]
        energy = energy*self.params_model[self.method]["energy_val2"] + self.params_model[self.method]["energy_val1"]*self.params_model[self.method]["LENGTH_SEQUENCE"]
        try:
            energy -= self.params_model[self.method]["EPS"]*self.params_model[self.method]["LENGTH_SEQUENCE"]
        except:
            pass
        return energy
    
    def denormalize_energy_tot_sep_model(self, energy, sep_model):
        model = self.params_model[self.method]["sep_models"][sep_model]
        if 'shift' in self.params_model[self.method]["ENERGY_NORM_TYPE"]:
            energy -= model["energy_val1"]/model["energy_val2"]*self.params_model[self.method]["LENGTH_SEQUENCE"]
        energy = energy*model["energy_val2"] + model["energy_val1"]*self.params_model[self.method]["LENGTH_SEQUENCE"]
        try:
            energy -= self.params_model[self.method]["EPS"]*self.params_model[self.method]["LENGTH_SEQUENCE"]
        except:
            pass
        return energy
    
    # Conversion from continuous to discrete:
    def cont2disc(self, s):
        (xc,yc,thc) = s
        xd = (np.floor((xc-(-self.map_size_x/2))/self.discr_x)).astype(int)
        yd = (np.floor((yc-(-self.map_size_y/2))/self.discr_y)).astype(int)
        if thc < -math.pi:
            thc += 2*math.pi
        elif thc > math.pi:
            thc -= 2*math.pi
        thd = (np.floor((thc-(-math.pi))/(self.discr_th))).astype(int)
        return (xd,yd,thd) 
    
    # Conversion from discrete to continuous    
    def disc2cont(self, s):
        (xd,yd,thd) = s
        xi = round(xd*self.discr_x- self.map_size_x/2, 4)
        yi = round(yd*self.discr_y - self.map_size_y/2, 4)
        th_i = round(thd*self.discr_th - math.pi, 4)
        return (xi, yi, th_i)
    
    def safety_check(self, xi_ref, yi_ref, yawi_ref):
        safe = True
        for xi,yi,yawi in zip(xi_ref, yi_ref, yawi_ref):
            p_wheels = self.all_wheels(xi,yi,yawi)
            for p_wheel in p_wheels:
                if abs(p_wheel[1].item()) >= self.map_size_y/2-0.5 or abs(p_wheel[0].item()) >= self.map_size_x/2-0.5:
                    safe = False
                    return safe
            if self.goal_type == "rectangle" and xi > self.end[0] + self.goal_depth/2:
                safe = False
                return safe
        return safe
    
    def goal_check(self, xi_ref, yi_ref, yawi_ref):
        next_state = (xi_ref[-1],yi_ref[-1],yawi_ref[-1])
        if self.goal_type == "circle":
            distance = np.sqrt(np.square(self.end[0]-next_state[0])+np.square(self.end[1]-next_state[1]))
            if distance < self.goal_radius:
                goal = True
            else:
                goal = False
        elif self.goal_type == "rectangle":
            if np.abs(self.end[0]-next_state[0]) <= self.goal_depth/2 and np.abs(self.end[1]-next_state[1]) <= self.goal_width/2:
                goal = True
            else:
                goal = False
        return goal
    
    
    def starting_pose(self, x0,y0,yaw0):
        # # Compute Wheels and Surrounding Position
        x_global = self.x_around_wheels*np.cos(yaw0) - self.y_around_wheels*np.sin(yaw0) + x0
        y_global = self.x_around_wheels*np.sin(yaw0) + self.y_around_wheels*np.cos(yaw0) + y0
        # Extract z points
        ix = np.floor((x_global-(-self.DEM_size_x*self.discr/2))/self.discr).astype(int)
        iy = np.floor((y_global-(-self.DEM_size_y*self.discr/2))/self.discr).astype(int)
        z = self.Z[ix,iy]
        
        # We select only 1 point for each wheel having max z
        points_z = []
        for zw in np.split(z,4):
            points_z.append(zw.max())
        z = np.array(points_z)       
        
        # Fit Plane from points
        A = np.c_[self.y_wheels_loc,self.x_wheels_loc,np.ones(self.y_wheels_loc.shape[0])]
        # A = np.c_[self.y_around_wheels,self.x_around_wheels,np.ones(self.y_around_wheels.shape[0])]
        fit,_,_,_ = np.linalg.lstsq(A, z, rcond=None)    # plane coefficients
        a = fit[0].item()
        b = fit[1].item()
        c = fit[2].item()
        # Compute Vectors in x and y directions
        z_vect = np.array([-a,-b,1])/np.linalg.norm([a,b,1])
        y_vect_y = math.sqrt(1/(1+(z_vect[0]/z_vect[2])**2))
        y_vect_z = -y_vect_y*z_vect[0]/z_vect[2]
        y_vect = np.array([y_vect_y, 0, y_vect_z])
        x_vect_x = math.sqrt(1/(1+(z_vect[1]/z_vect[2])**2))
        x_vect_z = -x_vect_x*z_vect[1]/z_vect[2]
        x_vect = np.array([0, x_vect_x, x_vect_z])
        # Compute roll and pitch angles
        roll0 = round(-np.arccos(np.dot(np.array([1,0,0]),y_vect))*180/math.pi,2)
        pitch0 = round(-np.arccos(np.dot(np.array([0,1,0]),x_vect))*180/math.pi,2)
        if y_vect_z>0:
            roll0 = -roll0
        if x_vect_z<0:
            pitch0 = -pitch0
        # Compute centre of robot elevation 
        dz = z-(a*self.y_wheels_loc + b*self.x_wheels_loc + c)
        z0 = round(c + dz.max(),3)+self.belly/np.cos(max(abs(roll0),abs(pitch0))*math.pi/180)
        return z0, roll0, pitch0
    
    def filling_missing(self, Z, valid=None):
        #mask valid values
        if valid is None:
            valid = ~np.isnan(Z)
        Z = interpolate.griddata((self.X_map[valid], self.Y_map[valid]), 
                                 Z[valid], (self.X_map,self.Y_map),  method='linear')
        valid = ~np.isnan(Z)
        if np.any(~valid):
            Z = interpolate.griddata((self.X_map[valid], self.Y_map[valid]), 
                                     Z[valid], (self.X_map,self.Y_map),  method='nearest')
        return Z
   
    
    #@ray.remote
    def wheel_trace(self, i, pi):
        (xi,yi,yawi) = pi
        
        X_global = self.X_local*np.cos(yawi) - self.Y_local*np.sin(yawi) + xi
        Y_global = self.X_local*np.sin(yawi) + self.Y_local*np.cos(yawi) + yi
        idx_x = (np.floor((X_global-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int)
        idx_y = (np.floor((Y_global-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)
        try:
            Z_map = self.Z[idx_x, idx_y]
        except:
            idx_x = np.clip(idx_x, 0, self.DEM_size_x)
            idx_y = np.clip(idx_y, 0, self.DEM_size_y)
            Z_map = self.Z[idx_x, idx_y]
        
        
        # First iteration I take all points from rear to front
        if not i:
            return list(Z_map[self.extra_eps//2:self.BASE_size+self.extra_eps//2, self.mask_map])
        else:
        # Other iterations I take only front 
            return Z_map[-self.extra_eps, self.mask_map]
    
    def segments_stats(self, data_run, ref_traj):
        dt = data_run.Time.values[1]-data_run.Time.values[0]
        stats = data_run.loc[data_run.Time >= self.params["run_delay"],:]
        xv, yv, yaw_v = ref_traj
        # Estimated Pitch and Roll from initial poisiton using point clouds
        pr_segments = self.estimate_pitch_roll(xv, yv, yaw_v)
        if 'conv1D' in self.method:
            W_trace = self.estimate_wheel_trace(xv, yv, yaw_v)
        # Splitting energy in segments from proprioceptive data
        stat_data = []
        power = np.array(stats["Motor_Speed"].values)*np.array(stats["Motor_Torque"].values)/1000   
        len_stats = len(stats)
        x_real, y_real, yaw_real = np.array(stats["X"].values),np.array(stats["Y"].values),np.array(stats["Yaw"].values)
        data_id = 0
        while data_id+1 < len_stats:
            xi,yi,yawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
            id_min = np.sqrt(np.square(xi-xv)+np.square(yi-yv)).argmin()
            segment = int(id_min//self.segment_points)
            # Measured Variables from proprioceptive data
            energy_segment = 0
            meas_pitches = []
            meas_rolls = []
            meas_speed_long = []
            meas_speed_lat = []
            # Loop last until new segment is detected (or end file)
            while data_id < len_stats:
                energy_segment += power[data_id]*dt
                meas_pitches.append(stats["Pitch"].values[data_id])
                meas_rolls.append(stats["Roll"].values[data_id])
                meas_speed_long.append(stats["FWD_Speed"].values[data_id])
                meas_speed_lat.append(stats["Lat_Speed"].values[data_id])
                if data_id+1<len_stats:
                    data_id += 1
                    xxi,yyi,yyawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
                    iid_min = np.sqrt(np.square(xxi-xv)+np.square(yyi-yv)).argmin()
                    ssegment = int(iid_min//self.segment_points)
                    if ssegment > segment:
                        if ssegment >= self.n_segments:
                            continue
                        else:
                            break
                else:
                    break
            stat_data.append({"energy": energy_segment,
                              "initial_speed_long": meas_speed_long[0], "mean_speed_long": np.mean(meas_speed_long),
                              "max_speed_long": np.max(meas_speed_long), "min_speed_long": np.min(meas_speed_long),
                              "initial_speed_lat": meas_speed_lat[0], "mean_speed_lat": np.mean(meas_speed_lat),
                              "max_speed_lat": np.max(meas_speed_lat), "min_speed_lat": np.min(meas_speed_lat),
                              "mean_pitch_meas": np.mean(meas_pitches), "var_pitch_meas": np.var(meas_pitches),
                              "mean_roll_meas": np.mean(meas_rolls), "var_roll_meas": np.var(meas_rolls),
                              "mean_pitch_est": pr_segments[segment,0], "mean_roll_est": pr_segments[segment,1],
                              "std_pitch_est": pr_segments[segment,2], "std_roll_est": pr_segments[segment,3]})                              
        
        if 'conv1D' in self.method:
           stat_data[-1]["wheel_trace_tot"] = W_trace
        
        return stat_data

    def estimate_pitch_roll(self, xi_ref, yi_ref, yawi_ref, remove_unpickable=True):
        
        # Estimated Pitch and Roll from initial poisiton using point clouds
        mean_pitch_segments = np.empty((self.n_segments))
        mean_roll_segments = np.empty((self.n_segments))
        std_pitch_segments = np.empty((self.n_segments))
        std_roll_segments = np.empty((self.n_segments))
        for segment in range(self.n_segments):
            start_id = segment*self.segment_points
            end_id = start_id + self.segment_points
            est_pitches = []
            est_rolls = []
            for xx,yy,yyaw in zip(xi_ref[start_id:end_id],yi_ref[start_id:end_id],yawi_ref[start_id:end_id]):
                _, est_roll, est_pitch = self.starting_pose(xx, yy, yyaw)
                est_pitches.append(est_pitch)
                est_rolls.append(est_roll)
            
            mean_pitch_segments[segment] = np.mean(est_pitches)
            mean_roll_segments[segment] = np.mean(est_rolls)
            std_pitch_segments[segment] = np.std(est_pitches)
            std_roll_segments[segment] = np.std(est_rolls)
        
        return np.stack([mean_pitch_segments,mean_roll_segments,std_pitch_segments,std_roll_segments], axis=-1)
        
    def estimate_wheel_trace(self, xi_ref, yi_ref, yawi_ref):
        
        zrel = self.Z[(np.floor((xi_ref[0]-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                      (np.floor((yi_ref[0]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]
        xi_ref, yi_ref, yawi_ref = xi_ref[:-1], yi_ref[:-1], yawi_ref[:-1]
        w_trace = [self.wheel_trace(i,pi) for i, pi in enumerate((zip(xi_ref, yi_ref, yawi_ref)))]        
        
        W_trace = np.array(w_trace[0] + w_trace[1:]) - zrel
            
        return np.round(W_trace,4)
    
    @ray.remote
    def merged_pitch_roll(self, state, action):
        xi, yi, yawi = state
        xi_ref, yi_ref, yawi_ref = self.action2traj(action, xi, yi, yawi, self.n_points)
        return self.estimate_pitch_roll(xi_ref, yi_ref, yawi_ref, remove_unpickable=False)
    
    @ray.remote
    def merged_wheel_trace(self, state, action):
        xi, yi, yawi = state
        xi_ref, yi_ref, yawi_ref = self.action2traj(action, xi, yi, yawi, self.n_points)
        return self.estimate_wheel_trace(xi_ref, yi_ref, yawi_ref, remove_unpickable=False)
        
    #@ray.remote
    def advance_state(self, ida, state):
        xi, yi, yawi = state
        xi_ref, yi_ref, yawi_ref = self.action2traj(ida, xi, yi, yawi, self.n_points)
        next_state = (xi_ref[-1],yi_ref[-1],yawi_ref[-1])
        # Check Safety
        safe = self.safety_check(xi_ref, yi_ref, yawi_ref)
        # Check Goal
        goal = self.goal_check(xi_ref, yi_ref, yawi_ref)
        
        output = {"safe": safe, "state": state, "next_state": next_state, "goal": goal, "action":ida}
        # Additional variables to compute cost
        if self.optimization_criteria == "distance":
            cost = self.length + abs(self.curvature_list[ida]) # discourage curves
        elif self.optimization_criteria == "pitch":
            pr = self.estimate_pitch_roll(xi_ref, yi_ref, yawi_ref)
            cost = max(-5,-np.mean(pr[:,0]))
        elif self.optimization_criteria == "energy":
            cost = None # estimate of energy is done in a following function to better exploit parallel computing
        output["cost"] = cost
        return output
    
    
    def create_shots_batch(self, n_shots = None):
        # Creating batch of past shots to inform the network
        # Note, shots are repeated if they are less than self.params_model["N_SHOTS"]
        if n_shots is None:
            n_shots = min(self.params_model[self.method]["N_SHOTS"],len(self.memory[self.method])//self.params_model[self.method]["LENGTH_SHOTS"])
        self.n_shots = n_shots
        if n_shots:
            # Most recent shots are used
            xe_shots = self.memory[self.method].loc[:,["energy"]][-n_shots*self.params_model[self.method]["LENGTH_SHOTS"]:].values
            if 'meta' in self.method:
                xe_shots = self.normalize_energy(xe_shots)
            if self.params_model[self.method]["SHOTS_EXTRA_INFO"]:
                xei_shots = self.memory[self.method].loc[:,self.params_model[self.method]["SHOTS_EXTRA_INFO"]][-n_shots*self.params_model[self.method]["LENGTH_SHOTS"]:].values
                xei_shots = self.normalize_extra_info(xei_shots)
                xe_shots = np.concatenate([xe_shots,xei_shots],axis=-1)
            if 'conv1D' in self.method:
                xg_shots = np.array(self.memory_geom_merged_list[self.method][-n_shots:])
                # Repeat shots if 0<n_shots<self.params_model["N_SHOTS"]
                for j in range(self.params_model[self.method]["N_SHOTS"]-n_shots):
                    xg_shots = np.concatenate((xg_shots,xg_shots[-1:]),axis=0)
                    xe_shots = np.concatenate((xe_shots,xe_shots[-self.params_model[self.method]["LENGTH_SHOTS"]:]))
            elif 'plane' in self.method:
                xg_shots = self.memory[self.method].loc[:,self.params_model[self.method]["INPUT_FEATURES"]][-n_shots*self.params_model[self.method]["LENGTH_SHOTS"]:].values
                # Repeat shots if 0<n_shots<self.params_model["N_SHOTS"]
                for j in range(self.params_model[self.method]["N_SHOTS"]-n_shots):
                    xg_shots = np.concatenate((xg_shots,xg_shots[-self.params_model[self.method]["LENGTH_SHOTS"]:]),axis=0)
                    xe_shots = np.concatenate((xe_shots,xe_shots[-self.params_model[self.method]["LENGTH_SHOTS"]:]))
                if 'meta' in self.method:
                    xg_shots = self.normalize_input_features(xg_shots)
                elif 'ST' in self.method:
                    xg_shots = xg_shots.reshape(self.params_model[self.method]["N_SHOTS"],self.params_model[self.method]["LENGTH_SEQUENCE"],len(self.params_model[self.method]["INPUT_FEATURES"]))
            
            # Repeat same shots for each element in the batch 
            # note: max num of element in a batch is equal to the num of actions in the lattice space  
            XG_SHOTS = np.stack([xg_shots for i in range(self.num_actions)],axis=0)
            XE_SHOTS = np.stack([xe_shots for i in range(self.num_actions)],axis=0)  
            return XG_SHOTS, XE_SHOTS
        else:
            return None, None
    
    def find_best_model(self):
        error = np.empty((len(self.model[self.method]),))
        XE_SHOTS = self.XE_SHOTS.reshape((self.XE_SHOTS.shape[0],self.XG_SHOTS.shape[1],self.params_model[self.method]["LENGTH_SEQUENCE"]))
        xe_shots = np.sum(XE_SHOTS,axis=-1)[0]
        for i, separate_model in enumerate(self.model[self.method]):
            if 'plane' in self.method:
                xg_shot = self.normalize_input_features_separate(self.XG_SHOTS[0], i)
            else:
                xg_shot = self.XG_SHOTS[0]
            ye = separate_model(xg_shot)
            ye = ye.numpy().squeeze(axis=-1)
            ye = self.denormalize_energy_tot_sep_model(ye,i)
            error[i] = np.sum(np.square(ye-xe_shots)).squeeze()
        return np.argmin(error)
        
    def separate_model_pred(self, XG_META):
        # Use it to make new prediction
        if 'plane' in self.method:
            xg_meta = self.normalize_input_features_separate(XG_META.squeeze(axis=1), self.best_m)
        else:
            xg_meta = XG_META.squeeze(axis=1)
        ye = (self.model[self.method][self.best_m](xg_meta)).numpy().squeeze(axis=-1)
        ye = self.denormalize_energy_tot_sep_model(ye,self.best_m)
        return ye
    
    def estimate_energy(self, outputs, remove_unpickable = True):
        bsize = len([1 for output in outputs if output["safe"]])
        if not bsize:
            return outputs
        # Creating batch of geometries of future trajectories  
        if self.n_shots:
            # keras model is not pickable by ray. We temporarly delete it
            if remove_unpickable:
                model = {}
                for method in self.all_methods:
                    model[method] = copy(self.model[method])
                del self.model
                
            if 'conv1D' in self.method:
                # Collect point cloud geometries of future trajectories
                # Very comput expensive in Conv1d. It is parallelised   
                #xg_meta = [self.merged_wheel_trace(output["state"],output["action"]) for output in outputs if output["safe"]]
                xg_meta = ray.get([self.merged_wheel_trace.remote(self, output["state"],output["action"]) for output in outputs if output["safe"]])
                XG_META = np.expand_dims(np.array(xg_meta),axis=1)
            elif 'plane' in self.method:
                xg_meta = ray.get([self.merged_pitch_roll.remote(self, output["state"],output["action"]) for output in outputs if output["safe"]])
                XG_META = np.array(xg_meta)
                if 'meta' in self.method:
                    XG_META = self.normalize_input_features(XG_META)
                elif 'ST' in self.method:
                    XG_META = XG_META.reshape(XG_META.shape[0],1,self.params_model[self.method]["LENGTH_SEQUENCE"],len(self.params_model[self.method]["INPUT_FEATURES"]))
            
             
            # restore keras models
            if remove_unpickable:
                self.model = {}
                for method in self.all_methods:
                    self.model[method] = copy(model[method])
            
            # Predicting energy. Note, the whole sequence is predicted, while the value for n_shots is at the n_shots-1 position
            if 'meta' in self.method:
                energies = (self.model[self.method]([self.XG_SHOTS[:bsize], self.XE_SHOTS[:bsize], XG_META])[self.n_shots-1]).numpy()
                energies = np.clip(self.denormalize_energy_tot(energies),0.0,None)
            elif "ST" in self.method:
                energies = self.separate_model_pred(XG_META)
        else:
            energies = np.zeros((bsize,))
            
        
        costs = energies
        
        id_e = 0
        for i,output in enumerate(outputs):
            if output["safe"]:
                outputs[i]["cost"] = costs[id_e].item()
                id_e += 1
        return outputs
        
    def min_goal(self, state):
        if self.goal_type == 'rectangle' and np.abs(self.end[1]-state[1]) <= self.goal_width/2:
            min_dist = np.abs(self.end[0]-self.goal_depth/2-state[0])
            min_zf = self.Z[(np.floor((self.end[0]-self.goal_depth/2-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                            (np.floor((state[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]
        else:
            min_dist = np.sqrt(np.square(self.end[0]-state[0])+np.square(self.end[1]-state[1]))
            min_zf = self.Z[(np.floor((self.end[0]-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                            (np.floor((self.end[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]  
        # min_dist = np.sqrt(np.square(self.end[0]-state[0])+np.square(self.end[1]-state[1]))
        # min_zf = self.Z[(np.floor((self.end[0]-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
        #                 (np.floor((self.end[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]  
        return min_dist, min_zf
        
    
    def heuristic(self, state):
        if self.optimization_criteria == "distance":
            distance = np.sqrt(np.square(self.end[0]-state[0])+np.square(self.end[1]-state[1]))
            if self.method != "fastest":
                return distance
            else:
                return distance*10            
        elif self.optimization_criteria == "energy" or self.optimization_criteria == "pitch":
            zs = self.Z[(np.floor((state[0]-(-self.DEM_size_x*self.discr/2))/self.discr)).astype(int),
                        (np.floor((state[1]-(-self.DEM_size_y*self.discr/2))/self.discr)).astype(int)]
            distance, zf = self.min_goal(state)
            if abs(distance) > self.discr:
                pitch = -np.arctan((zf-zs)/distance)*180/math.pi
            else:
                pitch = 0.0
            if self.optimization_criteria == "energy":
                if self.params_model[self.method]["HEURISTIC_TYPE"] == "opt":
                    energy = max(0,-1.41*pitch-4)*distance/self.segment_length
                return energy*self.meta_h_scaling
            elif self.optimization_criteria == "pitch":
                return max(0,-pitch*distance/self.length)
            
    
    def neighbours(self,state):
        state = self.disc2cont(state)
        if self.straight_flag:
            actions = [self.actions_sorted[0]]
        else:
            actions = self.actions_sorted
        # Check for goal, safety, and compute next state
        outputs = [self.advance_state(ida, state) for ida in actions]
        # Estimate energy cost
        if self.optimization_criteria == "energy":               
            outputs = self.estimate_energy(outputs)
        # Estimate heuristic cost   
        heuristics = [self.heuristic(output["next_state"]) if output["safe"] else None for output in outputs]
        for i in range(len(outputs)):
            outputs[i]["heuristic"] = heuristics[i]
        
        
        return outputs
        
                     
    def retrieve_solution(self, final_state):
        solution = {}
        for key in self.state_info[final_state].keys():
            solution[key] = []
        s = self.disc2cont(final_state)
        solution["state"].append((s[0], s[1], round(s[2]*180/math.pi,1)))
        while True:
            if self.state_info[final_state]["state"] is not None:
                for key,val in self.state_info[final_state].items():
                    if key == "state":
                        s = self.disc2cont(val)
                        solution[key].append((s[0], s[1], round(s[2]*180/math.pi,1)))
                        final_state = val
                    else:
                        solution[key].append(val)         
            else:
                break
        for key in solution.keys():
            solution[key].reverse()
        
        solution["elapsed_time"] = self.elapsed_time
        solution["nodes_expanded"] = self.nodes_expanded
        solution["evaluated_safe_branches"] = self.evaluated_safe_branches
        
        return solution
        
    def search(self, start, end, optimization_criteria = 'energy', method = "meta", straight_flag=False):
        self.straight_flag = straight_flag
        if optimization_criteria not in self.all_opt_criterias:
            print("Non-valid optimization criteria")
            print("Options: ", self.all_opt_criterias)
            return None
        self.optimization_criteria = optimization_criteria
        if method not in self.all_methods and method != "fastest":
            print("Non-valid method")
            print("Options: ", self.all_methods+" fastest")
            return None
        self.method = method
        
        self.OPEN = PriorityQueue()
        start = self.cont2disc(start)
        self.end = end
        self.OPEN.put(start,0)
        self.state_info = {}
        self.state_info[start] = {"state": None, "action": None, "goal": False, "cost": None}
        self.cost_so_far = {}
        self.cost_so_far[start] = 0
        
        if self.optimization_criteria == "energy":
            self.update_memory()
            # Retrieve available shots from previous observations
            print("Available shots: ", len(self.memory[self.method])//self.params_model[self.method]["LENGTH_SHOTS"])
            self.XG_SHOTS, self.XE_SHOTS = self.create_shots_batch()
            print("Using previous {}".format(self.n_shots))
            if self.n_shots and 'ST' in self.method:
                # Identify most similar model based on shots
                self.best_m = self.find_best_model()
                print("Best Separate Model: Terrain {}".format(self.params_model[self.method]["sep_models"][self.best_m]["id_t"]))
                  
            
        find = False
        self.elapsed_time = 0
        self.nodes_expanded = 0
        self.evaluated_safe_branches = 0
        exception_time = False
        period_count = 0
        
        self.start_time = time.time()
        while not self.OPEN.empty() and not find:
            current_state = self.OPEN.get()
            if self.state_info[current_state]["goal"]: # if goal is True
                find = True
                final_state = current_state
                break

            outputs = self.neighbours(current_state)
            self.nodes_expanded += 1
            for output in outputs:
                if output["safe"]:
                    self.evaluated_safe_branches += 1
                    new_cost = self.cost_so_far[current_state] + output["cost"]
                    next_state = self.cont2disc(output["next_state"])
                    if next_state not in self.cost_so_far or new_cost < self.cost_so_far[next_state]:
                        self.cost_so_far[next_state] = new_cost
                        if output["goal"]:
                            if self.method == "fastest":
                                find = True
                                final_state = next_state
                            priority = new_cost
                        else:
                            priority = new_cost + output["heuristic"]
                    
                        self.OPEN.put(next_state, priority)
                        self.state_info[next_state] = {}
                        for key, val in output.items():
                            if key == "state":
                                self.state_info[next_state][key] = current_state
                            else:
                                self.state_info[next_state][key] = val
            
            self.elapsed_time = time.time() - self.start_time
            if  self.elapsed_time > self.timeout:
                exception_time = True
                break
            elif self.elapsed_time >= self.log_time_period*period_count:
                period_count += 1
                if self.method != "fastest":
                    print("Elapsed Time: {}s. "
                          "Nodes Expanded: {}. "
                          "Elements in Queue: {}".format(round(self.elapsed_time,3),self.nodes_expanded,len(self.OPEN.elements)))
        if exception_time:
            if self.method != "fastest":
                print("Path planner run out of time! Retrieving best available solution...")
                # Look if a solution exists
                while not self.OPEN.empty():
                    current_state = self.OPEN.get()
                    if current_state in self.state_info:
                        if self.state_info[current_state]["goal"]:
                            find = True
                            final_state = current_state
                            break
        
        if self.method != "fastest":
            self.elapsed_time = time.time() - self.start_time
            print("Elapsed Time: {}s. "
                "Nodes Expanded: {}. "
                "Elements in Queue: {}".format(round(self.elapsed_time,3),self.nodes_expanded,len(self.OPEN.elements)))            
                
        if find:
            return self.retrieve_solution(final_state)
        else:
            # print("Path not found")
            return None
    
        


