# Dataset collection Augmented with obstacles. Obstacles are rigid (type 9)
import os
import numpy as np
from PIL import Image
import math
import random
import pandas as pd
import my_chrono_simulator as mcs
import terrain_generator as tg
import heapq
from datetime import datetime
from scipy import interpolate
import matplotlib.pyplot as plt
import argparse
import time

SEED = None

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--job', type=int, default=0, help='Job ID')
parser.add_argument('-p', '--path', default="", help='Path Experiment')
args = parser.parse_args()

VISUALISATION = False
plot_maps = False


terrain_ids = [12,8,7,0,22,5,14,13,4,9,11,10,1,3,15,16,17]
    
num_runs_per_id = 1500
simplex_terrain_types = ["wavy","smooth","rough"]
simplex_types_probs = [0.65,0.35,0.] # proportions of the terrain types in the dataset 
terrain_params_noise = 0 # noise to add to the terramechanical parameters (percentage of each value)


map_size_x = 30
map_size_y = 7
x0 = -12
y0 = 0
yaw0 = 0



# Augment with obstacles
AUGMENT_OBSTACLE = False
obst_per_unit_area = 0.4
# Position of obstacles (as percentage of the map size)
pos_obst_x_min = 0.05
pos_obst_x_max = 0.9
pos_obst_y_min = 0.1
pos_obst_y_max = 0.9
# Obstacles properties
obstacles = []
map_area_obst = map_size_x*map_size_y*(pos_obst_x_max-pos_obst_x_min)*(pos_obst_y_max-pos_obst_y_min)
n_obstacles = int(map_area_obst*obst_per_unit_area)
obstacles.append({"number": n_obstacles, "size": 6, "max_range": 0.15, "valley_prob": 0, "rigid_prob": 1})


target_speed = 1
run_delay = 0.5

random_action_sequence = True
sequence_length = 9
sequence_id_a = [2]*9

num_actions = 5
length = 2.7
max_angle = 20*math.pi/180

segment_length = 0.9
n_points = 49
n_segments = int(length/segment_length)
segment_points = int((n_points-1)//n_segments)

# Set Experiment directories and terrains
path_data = "./Datasets/"
if args.path:
    path_experiment = path_data + args.path
    path_maps = path_experiment + "Simplex_Maps/"
    terrain_ids_final = [terrain_ids[args.job]]
    path_output_file = path_experiment + "data_{}.csv".format(terrain_ids[args.job])
else:
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    path_experiment = path_data + "Exp_{}/".format(current_time)
    path_maps = path_experiment + "Simplex_Maps/"
    terrain_ids_final = terrain_ids
    path_output_file = path_experiment + "data.csv"


# Check if I am restarting a previous experiment
if os.path.exists(path_output_file):
    df = pd.read_csv(path_output_file)
    first_run = df.run.max() + 1
    random.seed(args.job+first_run)
    del df
else:
    first_run = 0
    random.seed(args.job)
    
if SEED:
    random.seed(SEED)
        

if args.job:
    # Job 0 creates all directories, the others wait for 2 seconds
    time.sleep(2)
else:
    # Create Directories
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    if not os.path.exists(path_experiment):
        os.mkdir(path_experiment)
    if not os.path.exists(path_maps):
        os.mkdir(path_maps)
    

# Constants not to change
# --------------------------------------------------------------------------------------------
belly = 0.5
wheelbase = 1.688965*2
wheeltrack = 0.95*2
eps_base = 0.5
eps_track = 0.54
discr = 0.0625
goal_radius = 0.5
y1_loc = -wheeltrack/2
x1_loc = wheelbase/2
y2_loc = wheeltrack/2
x2_loc = wheelbase/2
y3_loc = -wheeltrack/2
x3_loc = -wheelbase/2
y4_loc = wheeltrack/2
x4_loc = -wheelbase/2
DEM_size_x = int(map_size_x/discr +1)
DEM_size_y = int(map_size_y/discr +1)
DEM_Y_base = np.ceil((wheeltrack+eps_track)/discr).astype(int)
DEM_X_base = np.ceil((wheelbase+eps_base)/discr).astype(int)
x = np.linspace(-map_size_x/2,map_size_x/2,num=DEM_size_x)
y = np.linspace(-map_size_y/2,map_size_y/2,num=DEM_size_y)
Y , X = np.meshgrid(y,x)
y_base = np.linspace(-(wheeltrack+eps_track-discr)/2,(wheeltrack+eps_track-discr)/2,DEM_Y_base)
x_base = np.linspace(-(wheelbase+eps_base-discr)/2,(wheelbase+eps_base-discr)/2,DEM_X_base)
Y_base, X_base = np.meshgrid(y_base,x_base)
# Action space
forward_actions = []
curvature_list = np.linspace(-max_angle/length,max_angle/length,num_actions)
for curvature in curvature_list:
    if curvature:
        forward_actions.append((1/curvature,length*curvature*180/math.pi))
    else:
        forward_actions.append((length,0))
        
BASE_size = int((wheelbase+eps_base)/discr + 1)
TRACK_size = int((wheeltrack+eps_track)/discr + 1)
wheeltrace_cells = 20  # should be <= to: int((wheeltrack+eps_track)/discr +1)/2
WHEELTRACE_SHAPE = (segment_points+np.ceil((wheelbase+eps_base)/discr).astype(int)-1, wheeltrace_cells*2)
# --------------------------------------------------------------------------------------------


def fill_missing_value(Z_input, mask_missing, method):
    Z_input[mask_missing] = np.nan
    x = np.arange(0, Z_input.shape[1])
    y = np.arange(0, Z_input.shape[0])
    #mask invalid values
    array = np.ma.masked_invalid(Z_input)
    xx, yy = np.meshgrid(x, y)
    #get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]
    Z_filled = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy),  method=method)
    return Z_filled

def augment_with_obstacles(obst, Z, Z_obst = None):
    if Z_obst is None:
        Z_obst = np.zeros(Z.shape, dtype=np.int32)
    n_obstacles = obst["number"]
    size = obst["size"]
    max_range = obst["max_range"]
    valley_prob = obst["valley_prob"]
    rigid_prob = obst["rigid_prob"]
    
    if hasattr(size, "__iter__"):
        if len(size) == 2:
            size = random.randint(size[0],size[1])
        elif len(size) == 1:
            size = size[0]
        else:
            size = 6
    
    for i in range(n_obstacles):
        pos_x = random.randint(int(DEM_size_x*pos_obst_x_min)+size+1,int(DEM_size_x*pos_obst_x_max)-size-1)
        pos_y = random.randint(int(DEM_size_y*pos_obst_y_min)+size+1,int(DEM_size_y*pos_obst_y_max)-size-1)
        dz = np.empty((size,size))
        dz[:] = np.nan
        size_intorno = size*2+1
        intorno_centre = int((size_intorno-1)/2)
        Z_intorno = Z[pos_x-int((size_intorno-1)/2):pos_x+int((size_intorno-1)/2+1),
                      pos_y-int((size_intorno-1)/2):pos_y+int((size_intorno-1)/2+1)]
        
        if random.random() < valley_prob:
            valley = True
        else:
            valley = False
        
        pixels = random.choices(range(size**2),k=2)
        for pixel in pixels:
            yi = int(pixel%size)
            xi = int(pixel//size)
            if valley:
                dz[xi,yi] = random.uniform(-max_range/2,0)
            else:
                dz[xi,yi] = random.uniform(0,max_range)
        if size%2:
            Z_intorno[intorno_centre-int((size-1)/2):intorno_centre+int((size-1)/2+1),
                      intorno_centre-int((size-1)/2):intorno_centre+int((size-1)/2+1)] += dz
            if random.random() < rigid_prob:
                Z_obst[pos_x-int((size-1)/2):pos_x+int((size-1)/2+1),
                       pos_y-int((size-1)/2):pos_y+int((size-1)/2+1)] = int(1)
        else:
            Z_intorno[intorno_centre-int(size/2-1):intorno_centre+int(size/2+1),
                      intorno_centre-int(size/2-1):intorno_centre+int(size/2+1)] += dz
            if random.random() < rigid_prob:
                Z_obst[pos_x-int(size/2-1):pos_x+int(size/2+1),
                       pos_y-int(size/2-1):pos_y+int(size/2+1)] = int(1)
        mask_missing = np.isnan(Z_intorno)
        if sum(sum(mask_missing)):
            try:
                Z_intorno = fill_missing_value(Z_intorno, mask_missing, 'cubic')
            except:
                pass
            missing_mask2 = np.isnan(Z_intorno)
            if sum(sum(missing_mask2)):
                Z_intorno = fill_missing_value(Z_intorno,missing_mask2,'nearest')
        
        Z[pos_x-int((size_intorno-1)/2):pos_x+int((size_intorno-1)/2+1),
          pos_y-int((size_intorno-1)/2):pos_y+int((size_intorno-1)/2+1)] = Z_intorno

    return Z, Z_obst  
def bread_first_path(start):
    class PriorityQueue:
        def __init__(self):
            self.elements = []
        
        def empty(self):
            return len(self.elements) == 0
        
        def put(self, item, priority):
            heapq.heappush(self.elements, (priority, item))
        
        def get(self):
            return heapq.heappop(self.elements)[1]
    
    frontier = PriorityQueue()
    frontier.put(start,0)
    state_info = {}
    state_info[start] = (None,0,None)
        
    find = False
    actions = list(range(num_actions))
    while not frontier.empty() and not find:
        current_state = frontier.get()
        (xi,yi,yawi) = current_state
        (parent,distance,prev_ida) = state_info[current_state]
        
        random.shuffle(actions)
        
        for ida in actions:
            safe = True
            xi_ref, yi_ref, yawi_ref = action2traj(ida, xi, yi, yawi, n_points)
            for xxi, yyi, yyawi in zip(xi_ref, yi_ref, yawi_ref):
                p_wheels = all_wheels((xxi, yyi, yyawi), wheeltrack, wheelbase)
                for p_wheel in p_wheels:
                    if abs(p_wheel[0].item()) >= map_size_y/2-0.5 or abs(p_wheel[1].item()) >= map_size_x/2-0.5:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                next_state = (xi_ref[-1],yi_ref[-1],yawi_ref[-1])
                frontier.put(next_state,-(distance+1) + random.uniform(-.5,.5))
                state_info[next_state] = (current_state,distance+1,ida)
                if distance+1 == sequence_length:
                    find = True
                    final_state = next_state
                    break
    id_a = []          
    if find:
        while True:
            (parent,distance,prev_ida) = state_info[final_state]
            if parent is not None:
                id_a.append(prev_ida)
                final_state = parent
            else:
                break
        id_a.reverse()
    return id_a
        
def norm_angle(theta):
    if theta > math.pi:
        theta -= 2*math.pi
    elif theta < -math.pi:
        theta += 2*math.pi
    return theta

def action2traj(id_a,x0,y0,yaw0,n_cells):
    (r, dyaw) = forward_actions[id_a]
    dyaw = dyaw*math.pi/180
    yaw = np.linspace(yaw0,yaw0+dyaw,n_cells)
    yaw = np.array(list(map(norm_angle, yaw)))
    if dyaw:
        x = x0 - r*np.sin(yaw0) + r*np.sin(yaw);
        y = y0 + r*np.cos(yaw0) - r*np.cos(yaw);
    else:
        x = np.linspace(x0, x0 + r*np.cos(yaw0), n_cells);
        y = np.linspace(y0, y0 + r*np.sin(yaw0), n_cells);
    return x,y,yaw
def all_wheels(state, width, length):
    (XC,YC,Theta) = state
    pcenter = np.array([[YC],[XC]])
    rcenter = np.matrix(((np.cos(Theta), np.sin(Theta)), (-np.sin(Theta), np.cos(Theta))))
    pbl=pcenter+rcenter*np.array([[-width/2],[-length/2]])
    pbr=pcenter+rcenter*np.array([[width/2],[-length/2]])
    ptl=pcenter+rcenter*np.array([[-width/2],[length/2]])
    ptr=pcenter+rcenter*np.array([[width/2],[length/2]])
    return [pbl,pbr,ptl,ptr]

        
def starting_pose(x0,y0,yaw0,Z):
    # # Compute Wheels Position
    pcenter = np.array([[y0],[x0]])
    rcenter = np.matrix(((np.cos(yaw0), np.sin(yaw0)), (-np.sin(yaw0), np.cos(yaw0))))
    p1=pcenter+rcenter*np.array([[y1_loc],[x1_loc]])
    x1=p1[1].item()
    y1=p1[0].item()
    p2=pcenter+rcenter*np.array([[y2_loc],[x2_loc]])
    x2=p2[1].item()
    y2=p2[0].item()
    p3=pcenter+rcenter*np.array([[y3_loc],[x3_loc]])
    x3=p3[1].item()
    y3=p3[0].item()
    p4=pcenter+rcenter*np.array([[y4_loc],[x4_loc]])
    x4=p4[1].item()
    y4=p4[0].item()

    z_intorno = []
    for xi,yi in zip([x1,x2,x3,x4],[y1,y2,y3,y4]):
        ixl = [(np.floor((xi-(-DEM_size_x*discr/2))/discr)).astype(int)]
        iyl = [(np.floor((yi-(-DEM_size_y*discr/2))/discr)).astype(int)]
        
        ixl[0] = max(0,min(ixl[0],DEM_size_x -1))
        iyl[0] = max(0,min(iyl[0],DEM_size_y -1))
        
        ## Add some points in a 7x7 square
        for i in range(min(3,max(0,ixl[0]))):
            ixl.append(ixl[0]-i-1)
        for i in range(min(3,max(0,DEM_size_x -1-ixl[0]))):
            ixl.append(ixl[0]+i+1)
        for i in range(min(3,max(0,iyl[0]))):
            iyl.append(iyl[0]-i-1)
        for i in range(min(3,max(0,DEM_size_y -1-iyl[0]))):
            iyl.append(iyl[0]+i+1)
        zl = []
        for ix in ixl:
            for iy in iyl:
                zl.append(Z[ix,iy])
        z_intorno.append(np.array(zl).max())
    zt = np.array(z_intorno)
    ## Method3 points to fit
    points_x = [x1_loc, x2_loc, x3_loc, x4_loc]
    points_y = [y1_loc, y2_loc, y3_loc, y4_loc]
    points_z = z_intorno
    # Fit Plane which better approximates the points
    tmp_A = []
    tmp_b = []
    for i in range(len(points_x)):
        tmp_A.append([points_y[i], points_x[i], 1])
        tmp_b.append(points_z[i])
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b 
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
    zw1 = a*y1_loc + b*x1_loc + c
    zw2 = a*y2_loc + b*x2_loc + c
    zw3 = a*y3_loc + b*x3_loc + c
    zw4 = a*y4_loc + b*x4_loc + c
    zw = np.array([zw1,zw2,zw3,zw4])
    dz = zt-zw
    z0 = round(c + dz.max(),3)+belly/np.cos(max(abs(roll0),abs(pitch0))*math.pi/180)
    #z0 = round(c + dz.max(),3)+belly
    return z0, roll0, pitch0

def wheel_trace(xp,yp,thp, Z, Z_types, plot=False):    
    zrel = Z[(np.floor((xp[0]-(-DEM_size_x*discr/2))/discr)).astype(int),
             (np.floor((yp[0]-(-DEM_size_y*discr/2))/discr)).astype(int)]
    Z_trace = []
    Z_trace_types = []
    
    eps = 2
    for i, (xi,yi,yawi) in enumerate(zip(xp,yp,thp)):
        X_local = (X - xi)*np.cos(yawi) + (Y - yi)*np.sin(yawi)
        Y_local = -(X - xi)*np.sin(yawi) + (Y - yi)*np.cos(yawi)
        mask1 = abs(Y_local) < (wheeltrack+eps_track+discr*eps/2)/2
        mask2 = abs(X_local) < (wheelbase+eps_base+discr*eps/2)/2
        mask = mask1*mask2
    
        points_Z = Z[mask]-zrel
        points_Z_types = Z_types[mask]
        points_X_local = X_local[mask]
        points_Y_local = Y_local[mask]
    
        # Converting into Rectangular-image-like Z matrix
        Z_map = np.empty((BASE_size+eps, TRACK_size+eps))
        Z_map[:] = np.nan
        Z_map_types = np.empty((BASE_size+eps, TRACK_size+eps))
        Z_map_types[:] = np.nan
    
        # Filling Z matrix
        mask_x = np.floor((points_X_local-(-(BASE_size+eps)*discr/2))/discr).astype(int)
        mask_y = np.floor((points_Y_local-(-(TRACK_size+eps)*discr/2))/discr).astype(int)
        Z_map[mask_x,mask_y] = points_Z
        Z_map_types[mask_x,mask_y] = points_Z_types
    
        # Filling Missing values
        missing_mask = np.isnan(Z_map)
        if sum(sum(missing_mask)):
            Z_map_filled = fill_missing_value(Z_map,missing_mask,'linear')
            missing_mask2 = np.isnan(Z_map_filled)
            if sum(sum(missing_mask2)):
                Z_map_filled2 = fill_missing_value(Z_map_filled,missing_mask2,'nearest')
                Z_final = Z_map_filled2
            else:
                Z_final = Z_map_filled
        else:
            Z_final = Z_map
        
        missing_mask = np.isnan(Z_map_types)
        if sum(sum(missing_mask)):
            Z_final_types = fill_missing_value(Z_map_types,missing_mask,'nearest')
        else:
            Z_final_types = Z_map_types
        
        # First iteration I take all points from rear to front
        if not i:
            for j in range(eps//2,BASE_size+eps//2):
                Z_wheel_left = Z_final[j,eps//2:wheeltrace_cells+eps//2]
                Z_wheel_right = Z_final[j,-wheeltrace_cells-eps//2:-eps//2]
                Z_wheel_tot = np.concatenate([Z_wheel_left,Z_wheel_right])
                Z_trace.append(Z_wheel_tot)
                
                Z_wheel_left_types = Z_final_types[j,eps//2:wheeltrace_cells+eps//2]
                Z_wheel_right_types = Z_final_types[j,-wheeltrace_cells-eps//2:-eps//2]
                Z_wheel_tot_types = np.concatenate([Z_wheel_left_types,Z_wheel_right_types])
                Z_trace_types.append(Z_wheel_tot_types)
        else:
            # Taking wheel trace from front wheels
            Z_wheel_left = Z_final[-eps,eps//2:wheeltrace_cells+eps//2]
            Z_wheel_right = Z_final[-eps,-wheeltrace_cells-eps//2:-eps//2]
            Z_wheel_tot = np.concatenate([Z_wheel_left,Z_wheel_right])
            Z_trace.append(Z_wheel_tot)
            
            Z_wheel_left_types = Z_final_types[-eps,eps//2:wheeltrace_cells+eps//2]
            Z_wheel_right_types = Z_final_types[-eps,-wheeltrace_cells-eps//2:-eps//2]
            Z_wheel_tot_types = np.concatenate([Z_wheel_left_types,Z_wheel_right_types])
            Z_trace_types.append(Z_wheel_tot_types)
    
    Z_trace = np.array(Z_trace)
    Z_trace_types = np.array(Z_trace_types)
    
    return Z_trace, Z_trace_types, zrel
def segments_stats(data_run, ref_traj, Z, Z_types, executed_actions = 1):
    dt = data_run.Time.values[1]-data_run.Time.values[0]
    stats = data_run.loc[data_run.Time >= run_delay,:]
    xv, yv, yaw_v = ref_traj
    # Estimated Pitch and Roll from initial poisiton using point clouds
    est_pitch_segments, est_roll_segments, W_trace_segments, W_types_segments, zrel_segments = estimate_pitch_roll(xv, yv, yaw_v, Z, Z_types, executed_actions)
    # Splitting energy in segments from proprioceptive data
    energy_segments = np.zeros((n_segments*executed_actions))
    meas_speed_long_segments = np.zeros((4,n_segments*executed_actions))
    meas_speed_lat_segments = np.zeros((4,n_segments*executed_actions))
    meas_roll_segments = np.zeros((2,n_segments*executed_actions))
    meas_pitch_segments = np.zeros((2,n_segments*executed_actions))
    power = np.array(stats["Motor_Speed"].values)*np.array(stats["Motor_Torque"].values)/1000   
    len_stats = len(stats)
    x_real, y_real, yaw_real = np.array(stats["X"].values),np.array(stats["Y"].values),np.array(stats["Yaw"].values)
    data_id = 0
    while data_id+1 < len_stats:
        xi,yi,yawi = x_real[data_id],y_real[data_id],yaw_real[data_id]
        id_min = np.sqrt(np.square(xi-xv)+np.square(yi-yv)).argmin()
        segment = int(id_min//segment_points)
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
                ssegment = int(iid_min//segment_points)
                if ssegment > segment:
                    if ssegment >= n_segments*executed_actions:
                        continue
                    else:
                        break
            else:
                break
        energy_segments[segment] = energy_segment
        meas_speed_long_segments[0,segment] = meas_speed_long[0] # initial speed
        meas_speed_long_segments[1,segment] = np.mean(meas_speed_long) # mean speed
        meas_speed_long_segments[2,segment] = np.max(meas_speed_long) # max speed
        meas_speed_long_segments[3,segment] = np.min(meas_speed_long) # min speed
        meas_speed_lat_segments[0,segment] = meas_speed_lat[0] # initial speed
        meas_speed_lat_segments[1,segment] = np.mean(meas_speed_lat) # mean speed
        meas_speed_lat_segments[2,segment] = np.max(meas_speed_lat) # max speed
        meas_speed_lat_segments[3,segment] = np.min(meas_speed_lat) # min speed
        meas_pitch_segments[0,segment] = np.mean(meas_pitches)
        meas_pitch_segments[1,segment] = np.var(meas_pitches)
        meas_roll_segments[0,segment] = np.mean(meas_rolls)
        meas_roll_segments[1,segment] = np.var(meas_rolls)
        
    return energy_segments, est_pitch_segments, est_roll_segments, W_trace_segments, W_types_segments, zrel_segments, meas_pitch_segments, meas_roll_segments, meas_speed_long_segments, meas_speed_lat_segments


def estimate_pitch_roll(xi_ref, yi_ref, yawi_ref, Z, Z_types, executed_actions = 1):
    # Estimated Pitch and Roll from initial poisiton using point clouds
    mean_pitch_segments = np.zeros((n_segments*executed_actions))
    mean_roll_segments = np.zeros((n_segments*executed_actions))
    var_pitch_segments = np.zeros((n_segments*executed_actions))
    var_roll_segments = np.zeros((n_segments*executed_actions))
    W_trace_segments = np.zeros((n_segments*executed_actions,)+WHEELTRACE_SHAPE)
    W_types_segments = np.zeros((n_segments*executed_actions,)+WHEELTRACE_SHAPE)
    zrel_segments = np.zeros((n_segments*executed_actions))
    for segment in range(n_segments*executed_actions):
        est_pitches = []
        est_rolls = []
        start_id = segment*segment_points
        end_id = start_id + segment_points
        xi_ref_segment = xi_ref[start_id:end_id]
        yi_ref_segment = yi_ref[start_id:end_id]
        yawi_ref_segment = yawi_ref[start_id:end_id]
        
        # Estimated pitch and roll
        for xx,yy,yyaw in zip(xi_ref_segment,yi_ref_segment,yawi_ref_segment):
            _, est_roll, est_pitch = starting_pose(xx, yy, yyaw, Z)
            est_pitches.append(est_pitch)
            est_rolls.append(est_roll)
        mean_pitch_segments[segment] = np.mean(est_pitches)
        mean_roll_segments[segment] = np.mean(est_rolls)
        var_pitch_segments[segment] = np.var(est_pitches)
        var_roll_segments[segment] = np.var(est_rolls)
        
        # Full geometry tracy from point cloud
        W_trace_segments[segment], W_types_segments[segment], zrel_segments[segment] = wheel_trace(xi_ref_segment, yi_ref_segment, yawi_ref_segment, Z, Z_types)
    
    return (mean_pitch_segments, var_pitch_segments), (mean_roll_segments, var_roll_segments) , W_trace_segments, W_types_segments, zrel_segments

def plot_colormesh(Z, title):
    fig = plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize = 35)
    ax.tick_params(labelsize=40)
    ax.set_xlabel("Y [m]", fontsize = 35)
    ax.set_ylabel("X [m]", fontsize = 35)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    im = plt.imshow(Z, cmap='viridis', interpolation='nearest')
    ax.set_ylim(ax.get_ylim()[::-1])
    cb = fig.colorbar(im, ax =ax)
    cb.ax.tick_params(labelsize=40)
    cb.set_label("Z [m]", fontsize=35, rotation = 90, va= "bottom", labelpad = 32)
    plt.show()
    
def main():
    for terrain_id in terrain_ids_final:
        data_flag = False
        for run in range(first_run,num_runs_per_id):
            data = []
            print("Terrain id: {}, Run: {}".format(terrain_id,run))
            # Generate Simplex Map
            simplex_terrain_type = random.choices(simplex_terrain_types,weights=simplex_types_probs,k=1)[0]
            simplex = tg.OpenSimplex_Map(map_size_x, map_size_y, discr, terrain_type = simplex_terrain_type, plot = False)
            simplex.sample_generator(plot=plot_maps)
            Z = simplex.Z
            if AUGMENT_OBSTACLE:
                Z_obst = np.zeros(Z.shape)
                for obst in obstacles:
                    Z, Z_obst = augment_with_obstacles(obst, Z, Z_obst)
            else:
                Z_obst = np.zeros(Z.shape)
            minz = np.min(Z)      
            Z = Z-minz
            map_height = np.max(Z)
            # Map is saved as image
            path_image = path_maps+"{0:02d}_{1:04d}_{2:.3f}.bmp".format(terrain_id,run,map_height)
            Z_pixel = (Z/map_height*255).astype(np.uint8)
            im = Image.new('L', (Z_pixel.shape[1],Z_pixel.shape[0]))
            im.putdata(Z_pixel.reshape(Z_pixel.shape[0]*Z_pixel.shape[1]))
            im = im.rotate(90, expand=True)
            im = im.resize((int(DEM_size_x/1),int(DEM_size_y/1)), Image.BILINEAR)
            im.save("{}".format(path_image)) 
            
            im = im.rotate(-90, expand=True)
            Z_pixel = np.array(list(im.getdata(0))).reshape((im.size[1],im.size[0]))
            Z = Z_pixel.astype(np.float32)/255.0*map_height
            
            if plot_maps:
                simplex.plot_colormesh(Z)  
            # Compute Initial Position
            z0, roll0, pitch0 = starting_pose(x0, y0, yaw0*math.pi/180, Z)
            prev_curv = 0
            # Define Action Sequence
            xi, yi, yawi = x0,y0,yaw0*math.pi/180
            if random_action_sequence:
                id_a = bread_first_path((xi,yi,yawi))
            else:
                id_a = sequence_id_a
                
            # Initialise Simulator
            sim = mcs.simulator(path_image, Z_obst, (map_size_x,map_size_y,map_height.item()), (x0,y0,z0), (roll0,pitch0,yaw0), terrain_id, terrain_params_noise, visualisation = VISUALISATION)
            sim.delay = run_delay
            # Execute Actions
            for i, ida in enumerate(id_a):
                # Define points for path follower
                xv,yv,yaw_v = action2traj(ida, xi, yi, yawi, n_points)
                zv = [z0]*len(xv)
                # Run simulator
                if not sim.run((xv,yv,zv), target_speed):
                    print("Failure")
                    break
                # Retrieving statistics from executed action and adding to data
                energy_segments, est_pitch_segments, est_roll_segments, W_trace_segments, W_types_segments, zrel_segments, meas_pitch_segments, meas_roll_segments, meas_speed_long_segments, meas_speed_lat_segments  = segments_stats(sim.data_run, (xv,yv,yaw_v), Z, Z_obst)
                curv = curvature_list[ida]
                for segment in range(len(energy_segments)):
                    if not segment:
                        curv_tm1 = prev_curv
                    else:
                        curv_tm1 = curv
                    W_string = ' '.join(str(round(v,4)) for v in W_trace_segments[segment].flatten())
                    W_string_types = ' '.join(str(np.int8(v)) for v in W_types_segments[segment].flatten())
                    data.append({"terrain_id": terrain_id, "run": run,
                                 "segment": int(segment+i*n_segments), 
                                 "curvature": curv, "curvature_tm1": curv_tm1,
                                 "energy": energy_segments[segment],
                                 "mean_pitch_est": est_pitch_segments[0][segment], "mean_roll_est": est_roll_segments[0][segment],
                                 "mean_pitch_meas": meas_pitch_segments[0,segment], "mean_roll_meas": meas_roll_segments[0,segment],
                                 "var_pitch_est": est_pitch_segments[1][segment], "var_roll_est": est_roll_segments[1][segment],
                                 "var_pitch_meas": meas_pitch_segments[1,segment], "var_roll_meas": meas_roll_segments[1,segment],
                                 "initial_speed_long": meas_speed_long_segments[0,segment], "mean_speed_long": meas_speed_long_segments[1,segment],
                                 "max_speed_long": meas_speed_long_segments[2,segment], "min_speed_long": meas_speed_long_segments[3,segment],
                                 "initial_speed_lat": meas_speed_lat_segments[0,segment], "mean_speed_lat": meas_speed_lat_segments[1,segment],
                                 "max_speed_lat": meas_speed_lat_segments[2,segment], "min_speed_lat": meas_speed_lat_segments[3,segment],
                                 "wheel_trace": W_string, "zrel": zrel_segments[segment]})
                    data_flag = True
                    if AUGMENT_OBSTACLE:
                        data[-1]["wheel_types"] = W_string_types
                        
                    if plot_maps:
                        plot_colormesh(W_trace_segments[segment], "Segment: {}".format(segment))
                # Next initial state
                xi, yi, yawi = xv[-1], yv[-1], yaw_v[-1]
                prev_curv = curv
                
                if plot_maps:
                    W_trace_tot = W_trace_segments[0]
                    zrel_0 = zrel_segments[0]
                    for p in range(1, W_trace_segments.shape[0]):
                        W_trace_tot = np.concatenate([W_trace_tot,W_trace_segments[p,-segment_points:]+zrel_segments[p]-zrel_0], axis = 0)
                    plot_colormesh(W_trace_tot, "Merged")
            
            # Close simulation
            sim.close()
            # Save data in memory
            if not run or not data_flag:
                pd.DataFrame(data).to_csv(path_output_file, index=False)
            else:
                pd.DataFrame(data).to_csv(path_output_file, mode='a', header=False, index=False)
            
            
        
            

if __name__ == "__main__":
    main()

    
    
    





