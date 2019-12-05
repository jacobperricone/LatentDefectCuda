#### READ ME ####
To fetch the data run the scrip get_data.sh. It should pull four files
Product_Small.txt, tfdf_Small.txt,Company_Small.txt,Issue_Small.txt and host them in the 
directory with main.cu and auxiliary.cu 

#### INPUTS ####
The code can take two command line arguments for the log file and results file names.

#### OUTPUTS ####
The code outputs two log files


### Limitations ###
The code as is runs the eStep kernel in place of the eStep2 kernel. To change which
implementation change the kernel call on line 982 of auxiliary.cu. There is extra code here 
to run the cuBLAS version of the M-STEP and also additional error checking code. I apologize
for not including more flexibility, but time constraints are limiting.


