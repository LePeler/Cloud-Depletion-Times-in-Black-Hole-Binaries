# Cloud-Depletion-Times-in-Black-Hole-Binaries #
Hi, my name is Fabian Wohlfahrt, this repository includes the python scripts I coded for my bachelors thesis in physics. 
You will find 3 scripts.
Before running the scripts, you need to create a folder to which all the data will be saved.
It needs to contain 3 subfolders: /Depletion-Data , /Dynamic-Data and /Results .
You then need to insert the folder path in the path variable at the beginning of all three programs.
# Simulator #
This script is the one, that actually runs the main simulations. 
It can be run from a terminal window, where you need to provide the system parameters, that you can find in the -help menu.
It will output two files with the important data for the analysis.
The file Data,{params}.npy saves the number of cloud particles around the lighter BH, it will be saved to the /Depletion-Data subfolder.
The file DData,{params}.npy saves the dynamic data of the BHs, it will be saved to the /Dynamic-Data subfolder.
# Data-Analysis #
This program will import all the data from the /Depletion-Data and /Dynamic-Data folders and extracts the depletion time from them. It then saves the results to the /Results folder.
It will print a message in the console for every analysed file stating, wheter the analysis worked, or not (sometimes the Newton method does not converge properly).
The errors can be solved by playing around with the variable N in line 63, varying it between 14990-15010 should do the trick.
# Data-Summary #
This program imports all the depletion times from the /Results folder and gives you functions to plot 2d slices of the data and perform fits on them,
as well as a function, which lets you fit to the entire dataset.
