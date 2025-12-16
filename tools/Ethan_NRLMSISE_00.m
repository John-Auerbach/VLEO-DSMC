clc
clear
close all

%% Purpose

% Ethan Kravet
% 13 August 2024

% Code developed to graph the atmospheric conditions as a fucntion of
% altittude, solar activity, etc. using NRLMSISE-00 and NRLMSISE 2.0 model.

%% Data Analysis

% Solar cycle #24, min Dec 2008, max Apr 2014, mean June 2011

alt = linspace(70e3,300e3,501);
f107a = 250;
f107 = 250;
aph = [4 4 4 4 4 4 4];
[T, rho] = atmosnrlmsise00(alt, 40.7934, -77.8600, 2011, 100, 0, f107a, f107, aph);

G = 6.6743*10^-11; % Gravitational constant
M = 5.97219*10^24; % Mass of Earth, kg
rE = 6378e3; % Radius of Earth, km
v_mps = sqrt(G*M./(rE+alt)); % m/s

p_dyn = (1/2) .* rho(:,6) .* (v_mps'.^2); % dynamic pressure, Pa

figure(1)
plot(alt/1000,rho(:,6))
xlabel('Altitude [km]','Interpreter','latex')
ylabel('Density [kg/m$^3$]','Interpreter','latex')
set(gcf,'color','w')

figure(2)
plot(alt/1000,T(:,2))
xlabel('Altitude [km]','Interpreter','latex')
ylabel('Temperature [K]','Interpreter','latex')
set(gcf,'color','w')

figure(3)
plot(alt/1000,v_mps)
xlabel('Altitude [km]','Interpreter','latex')
ylabel('Orbital Velocity [m/s]','Interpreter','latex')
set(gcf,'color','w')

figure(4)
plot(alt/1000,p_dyn)
xlabel('Altitude [km]','Interpreter','latex')
ylabel('Dynamic Pressure [Pa]','Interpreter','latex')
set(gcf,'color','w')

figure(5)
plot(alt/1000,p_dyn*0.04)
xlabel('Altitude [km]','Interpreter','latex')
ylabel('Drag [N]','Interpreter','latex')
set(gcf,'color','w')

headers = {'Altitude [km]','He [1/m^3]','O [1/m^3]','N2 [1/m^3]','O2 [1/m^3]',...
    'Ar [1/m^3]','Mass Density [kg/m^3]','H [1/m^3]','N [1/m^3]',...
    'Anomalous O [1/m^3]','Exospheric Temp [K]','Temp at Altitude [K]'};
all_data = [alt'/1000, rho, T];
all_data_table = array2table(all_data, 'VariableNames', headers);
writetable(all_data_table,'NRLMSISE00 Data.xlsx')

%% Using atmosnrlmsise00

% atmosnrlmsise00 Use NRLMSISE-00 atmosphere model.
%  [T, RHO] = atmosnrlmsise00( H, LAT, LON, YEAR, DOY, SEC, LST, F107A,
%  F107, APH, FLAGS, ITYPE, OTYPE, ACTION ) implements the mathematical
%  representation of the 2001 United States Naval Research Laboratory Mass
%  Spectrometer and Incoherent Scatter Radar Exosphere (NRLMSISE-00),
%  of the MSIS(R) class model. NRLMSISE-00 calculates the neutral
%  atmosphere empirical model from the surface to lower exosphere (0 to
%  1,000,000 meters) with the option of including contributions from
%  anomalous oxygen which can affect satellite drag above 500,000 meters. 
% 
%  Inputs for atmosnrlmsise00 are:
%  H      :an array of M altitude in meters.
%  LAT    :an array of M geodetic latitude in degrees.
%  LON    :an array of M geodetic longitude in degrees.
%  YEAR   :an array of M year. Year is currently ignored in this model. 
%  DOY    :an array of M day of year. Day of year ranges from 1 to 365 (or
	%        366).
%  SEC    :an array of M seconds in day in universal time (UT)
%  LST    :an array of M local apparent solar time (hours). To obtain a
%         physically realistic value, LST is set to (SEC/3600 + LON/15) by
%         default.  See Limitation section for more information.
%  F107A  :an array of M 81 day average of F10.7 flux (centered on doy).
%         If F107A is input, F107 and APH must also be input. The effects
%         of F107A are neither large nor well established below 80,000
%         meters, therefore the default value is set to 150. See
%         Limitation section for more information. 
%  F107   :an array of M daily F10.7 flux for previous day. If F107 is
%         input, F107A and APH must also be input. The effects of F107 are
%         neither large nor well established below 80,000 meters,
%         therefore the default value is set to 150. See Limitation
%         section for more information.
%  APH    :an array of M-by-7 of magnetic index information. If APH is
%         input, F107A and F107 must also be input. This information
%         consists of daily magnetic index (AP), 3 hour AP for current
%         time, 3 hour AP for 3 hours before current time, 3 hour AP for 6
%         hours before current time, 3 hour AP for 9 hours before current
%         time, average of eight 3 hour AP indices from 12 to 33 hours
%         prior to current time, and average of eight 3 hour AP indices
%         from 36 to 57 hours prior to current time. The effects of daily
%         magnetic index are neither large nor well established below
%         80,000 meters, therefore the default value is set to 4. See
%         Limitation section for more information.  
%  FLAGS  :a numerical array of 23 values for setting particular
%         variations in calculation the output.  Setting a value to 0.0
%         removes that value's effect on the output.  Setting a value to
%         1.0 applies the main and the cross term effects of that value
%         on the output.  Setting a value to 2.0 applies only the cross
%         term effect of that value on the output.  Additionally setting
%         FLAGS(9) = -1 uses the entire matrix APH rather than just
%         APH(:,1). The variations contained in FLAGS are ordered as
%         follows: 
%          FLAGS(1)  :F10.7 effect on mean  
%          FLAGS(2)  :Time independent
%          FLAGS(3)  :Symmetrical annual    
%          FLAGS(4)  :Symmetrical semi-annual
%          FLAGS(5)  :Asymmetrical annual   
%          FLAGS(6)  :Asymmetrical semi-annual
%          FLAGS(7)  :Diurnal               
%          FLAGS(8)  :Semi-diurnal
%          FLAGS(9)  :Daily AP             
%          FLAGS(10) :All UT, longitudinal effects
%          FLAGS(11) :Longitudinal        
%          FLAGS(12) :UT and mixed UT, longitudinal
%          FLAGS(13) :Mixed AP, UT, longitudinal     
%          FLAGS(14) :Ter-diurnal
%          FLAGS(15) :Departures from diffusive equilibrium
%          FLAGS(16) :All exospheric temperature variations         
%          FLAGS(17) :All variations from 120,000 meter temperature (TLB)
%          FLAGS(18) :All lower thermosphere (TN1) temperature variations           
%          FLAGS(19) :All 120,000 meter gradient (S) variations
%          FLAGS(20) :All upper stratosphere (TN2) temperature variations           
%          FLAGS(21) :All variations from 120,000 meter values (ZLB)
%          FLAGS(22) :All lower mesosphere temperature (TN3) variations           
%          FLAGS(23) :Turbopause scale height variations
%         The default values are 1.0 for all FLAGS.
%  OTYPE  :a string specifying if the total mass density output will
%         include anomalous oxygen ('Oxygen') or not ('NoOxygen'). The
%         default is 'NoOxygen'.
%  ACTION :a string to determine action for out-of-range input. Specify if
%         out-of-range input invokes a 'Warning', 'Error', or no action
%         ('None'). The default is 'Warning'.
% 
%  Outputs calculated for the NRLMSISE-00 model are: 
%  T      :an array of M-by-2 values of temperatures.  These values are
%          exospheric temperature in Kelvin and temperature at altitude in
%          Kelvin.
%  RHO    :an array of M-by-9 values of densities.  These values are
%          HE number density in meters^-3, O number density in meters^-3,
%          N2 number density in meters^-3, O2 number density in meters^-3,
%          AR number density in meters^-3, total mass density in kilogram 
%          per meters cubed, H number density in meters^-3, N number
%          density in meters^-3, and Anomalous oxygen number density in
%          meters^-3. 
% 
%  Limitation:
% 
%  If array length, M, is 23 and all available inputs are not specified,
%  FLAGS will always be assumed to be set.
% 
%  This function has the limitations of the NRLMSISE-00 model. For more
%  information see the documentation. 
% 
%  SEC, LST, and LON are used independently in the NRLMSISE-00 model and
%  are not of equal importance for every situation. For the most
%  physically realistic calculation these three variables are chosen to be
%  consistent by default (LST = SEC/3600 + LON/15). Departures from the
%  prior formula for LST can be included if available but are of minor
%  importance.
% 
%  F107 and F107A values used to generate the model correspond to the 10.7
%  cm radio flux at the actual distance of the Earth from the Sun rather
%  than the radio flux at 1 AU. The following site provides both classes
%  of values:
%  https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/noontime-flux/penticton/
% 
%  Examples:
% 
%  Calculate the temperatures, densities not including anomalous oxygen
%  using NRLMSISE-00 model at 10000 meters, 45 degrees latitude, -50
%  degrees longitude, on January 4, 2007 at 0 UT using default values for
%  flux, magnetic index data, and local solar time with out-of-range
%  actions generating warnings:    
%     [T, rho] = atmosnrlmsise00( 10000, 45, -50, 2007, 4, 0)
% 
%  Calculate the temperatures, densities not including anomalous oxygen
%  using NRLMSISE-00 model at 10000 meters, 45 degrees latitude, -50
%  degrees longitude, and at 25000 meters, 47 degrees latitude, -55
%  degrees longitude on January 4, 2007 at 0 UT using default values for
%  flux, magnetic index data, and local solar time with out-of-range
%  actions generating warnings:    
%     [T, rho] = atmosnrlmsise00( [10000; 25000], [45; 47], [-50; -55], [2007; 2007], [4; 4], [0; 0] )
% 
%  Calculate the temperatures, densities including anomalous oxygen
%  using NRLMSISE-00 model at 10000 meters, 45 degrees latitude, -50
%  degrees longitude, on January 4, 2007 at 0 UT using default values for
%  flux, magnetic index data, and local solar time with out-of-range
%  actions generating errors:    
%     [T, rho] = atmosnrlmsise00( 10000, 45, -50, 2007, 4, 0, 'Oxygen', 'Error' )
% 
%  Calculate the temperatures, densities including anomalous oxygen
%  using NRLMSISE-00 model at 100000 meters, 45 degrees latitude, -50
%  degrees longitude, on January 4, 2007 at 0 UT using defined values for
%  flux, and magnetic index data, and default local solar time with out of
%  range actions generating no message:    
%     aph = [17.375 15 20 15 27 (32+22+15+22+9+18+12+15)/8 (39+27+9+32+39+9+7+12)/8]
%     f107 = 87.7
%     nov_6days  = [78.6 78.2 82.4 85.5 85.0 84.1]
%     dec_31daymean = 84.5
%     jan_31daymean = 83.5
%     feb_13days = [89.9 90.3 87.3 83.7 83.0 81.9 82.0 78.4 76.7 75.9 74.7 73.6 72.7]
%     f107a = (sum(nov_6days) + sum(feb_13days) + (dec_31daymean + jan_31daymean)*31)/81
%     flags = ones(1,23)
%     flags(9) = -1
%     [T, rho] = atmosnrlmsise00( 100000, 45, -50, 2007, 4, 0, f107a, f107, aph, flags, 'Oxygen', 'None' )