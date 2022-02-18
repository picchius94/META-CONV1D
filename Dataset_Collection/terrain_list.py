def Soil_Params(terrain_id):
    
    # Params are: Bekker Kphi, Bekker Kc, Bekker n exponent,
    #             Mohr cohesive limit (Pa), Mohr friction limit (degrees),
    #             Janosi shear coefficient (m),
    #             Elastic stiffness (Pa/m), before plastic yield, must be > Kphi,
    #             Damping (Pa s/m), proportional to negative vertical speed (optional)
    
    # Dry Sand (Land Locomotion Lab, LLL) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    if terrain_id == 0:
        return [1528.43e3, 0.99e3, 1.1, 1.04e3, 28, 2e-2, 4e8, 3e4]
    # Sandy Loam (15% moisture content) (Land Locomotion Lab, LLL) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 1:
        return [1515.04e3, 5.27e3, 0.7, 1.72e3, 29, 2e-2, 4e8, 3e4]
    # Sandy Loam (22% moisture content) (Land Locomotion Lab, LLL) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 2:
        return [43.12e3, 2.56e3, 0.2, 1.38e3, 38, 2e-2, 4e8, 3e4]
    # Sandy Loam (11% moisture content) (Michigan, Strong, Buchele) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 3:
        return [1127.97e3, 52.53e3, 0.9, 4.83e3, 20, 2e-2, 4e8, 3e4]
    # Sandy Loam (23% moisture content)(Michigan, Strong, Buchele) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 4:
        return [808.96e3, 11.42e3, 0.4, 9.65e3, 35, 2e-2, 4e8, 3e4]
    # Sandy Loam (26% moisture content) (Hanamoto) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 5:
        return [141.11e3, 2.79e3, 0.3, 13.79e3, 22, 2e-2, 4e8, 3e4]
    # Sandy Loam (32% moisture content) (Hanamoto) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 6:
        return [51.91e3, 0.77e3, 0.5, 5.17e3, 11, 2e-2, 4e8, 3e4]
    # Clayey Soil (38% moisture content) (Thailand) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 7:
        return [692.15e3, 13.19e3, 0.5, 4.14e3, 13, 2e-2, 4e8, 3e4]
    # Clayey Soil (55% moisture content) (Thailand) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 8:
        return [1262.53e3, 16.03e3, 0.7, 2.07e3, 10, 2e-2, 4e8, 3e4]
    # Heavy Clay (25% moisture content) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 9:
        return [1555.95e3, 12.7e3, 0.13, 68.95e3, 34, 0.6e-2, 4e8, 3e4]
    # Heavy Clay (40% moisture content) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 10:
        return [103.27e3, 1.84e3, 0.11, 20.69e3, 6, 3e-2, 4e8, 3e4]
    # Lean Clay (22% moisture content) (WES) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 11:
        return [1724.69e3, 16.43e3, 0.2, 68.95e3, 20, 0.6e-2, 4e8, 3e4]
    # Lean Clay (32% moisture content) (WES) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 12:
        return [119.61e3, 1.52e3, 0.15, 13.79e3, 11, 3e-2, 4e8, 3e4]
    # LETE sand (Wong) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 13:
        return [5301e3, 102e3, 0.79, 1.3e3, 31.1, 1.2e-2, 4e8, 3e4]
    # Upland sandy loam (51% moisture content) (Wong) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 14:
        return [2080e3, 74.6e3, 1.10, 3.3e3, 33.7, 2e-2, 4e8, 3e4]
    # Rubicon sandy loam (43% moisture content) (Wong) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 15:
        return [752e3, 6.9e3, 0.66, 3.7e3, 29.8, 2e-2, 4e8, 3e4]
    # North Gower clayey loam (46% moisture content) (Wong) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 16:
        return [2471e3, 41.6e3, 0.73, 6.1e3, 26.6, 2e-2, 4e8, 3e4]
    # Grenville loam (24% moisture content) (Wong) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 17:
        return [5880e3, 0.06e3, 1.01, 3.1e3, 29.8, 2e-2, 4e8, 3e4]
    # Snow (U.S.) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 18:
        return [196.72e3, 4.37e3, 1.6, 1.03e3, 19.7, 3e-2, 4e8, 3e4]
    # Snow (U.S., Harrison) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 19:
        return [245.90e3, 2.49e3, 1.6, 0.62e3, 23.2, 3e-2, 4e8, 3e4]
    # Snow (Sweden) parameters (but last 3) from "Theory of Ground Vehicles", Wong et al., 2008
    elif terrain_id == 20:
        return [66.08e3, 10.55e3, 1.44, 6e3, 20.7, 3e-2, 4e8, 3e4]
    # DLR Mechanical Mars soil simulant (DLR), from "Strenght of soil deposit MER Traverses", NASA
    elif terrain_id == 21:
        return [180e3, 5.79e3, 0.8, 0.3e3, 17.8, 1e-2, 4e8, 3e4]
    # Lunar Nominal Soil, from "Strenght of soil deposit MER Traverses", NASA
    elif terrain_id == 22:
        return [820e3, 1.4e3, 1, 0.17e3, 35, 1e-2, 4e8, 3e4]
    else:
        return None