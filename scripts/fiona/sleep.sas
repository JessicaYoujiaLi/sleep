PROC IMPORT FILE="C:\Users\16124\Desktop\sleep\sleep.csv"
	OUT=sleep
	DBMS=csv
	REPLACE;
RUN;

data sleep;
   set sleep;
   roi_label_cha=put(roi_label,8.);
 run;
 
proc contents data=sleep;
run;

proc glimmix data=sleep;
	class roi_label_cha;
   model dfof = sleep_numeric / solution;
   random roi_label_cha / type=ar(1) residual;
run;

proc mixed data=sleep;
	class roi_label_cha;
   model dfof = sleep_numeric / solution;
   random roi_label_cha / type=ar(1);
run;

proc mixed data=sleep;
	class roi_label_cha;
   model dfof = sleep_numeric / solution;
   random intercept /subject = roi_label_cha;
run;

*plot;
proc sgplot data=sleep;
   vbox dfof / group=sleep_numeric;
run;

proc mixed data=sleep;
   class roi_label_cha;
   model dfof = sleep_numeric / solution;
   random intercept /subject = roi_label_cha;
   estimate 'NREM vs Awake' sleep_numeric 1 -1 / cl;
run;

/* create a dataset with the estimated marginal means */
proc means data=sleep noprint;
   class sleep_numeric;
   output out=means mean=dfof_mean;
run;

/* create a new variable to indicate if NREM > Awake */
data means;
   set means;
   if sleep_numeric = 1 then nrem_gt_awake = (dfof_mean > lag(dfof_mean));
run;

*spikes;
PROC IMPORT FILE="C:\Users\16124\Desktop\sleep\spike_clean.csv"
	OUT=spike
	DBMS=csv
	REPLACE;
RUN;

data spike;
   set spike;
   roi_label_cha=put(roi_label,8.);
 run;

proc freq data = spike;
table spikes;
run;

data spike;
set spike;
if spikes = 0 then spike_binary = 0;
else spike_binary = 1;
run;

proc glimmix data=spike;
  class roi_label_cha;
  model spike_binary(event='1') = sleep_numeric / solution;
  random roi_label_cha / subject=roi_label_cha type=un;
run;

*cell;

PROC IMPORT FILE="C:\Users\16124\Desktop\sleep\cell2.csv"
	OUT=cells
	DBMS=csv
	REPLACE;
RUN;

data cells;
   set cells;
   roi_label_cha=put(roi_label,8.);
 run;
 
proc freq data = cells;
table sleep_numeric;
run;
 
*good;
 proc mixed data=cells;
	class roi_label_cha;
   model dfof = cell_numeric/ solution;
   random  intercept /subject =roi_label_cha;
run;

 proc mixed data=cells;
	class roi_label_cha;
   model dfof = cell_numeric sleep_numeric cell_numeric*sleep_numeric/ solution;
   random  intercept /subject =roi_label_cha;
run;

*plot;

proc sgplot data=cells;
   vbox dfof / category=sleep_numeric group=cell_numeric;
run;


proc glimmix data=cells;
	class roi_label_cha;
   model dfof = cell_numeric/ solution;
   random roi_label_cha / type=ar(1) residual;
run;

 proc mixed data=cells;
	class roi_label_cha;
   model dfof = cell/ solution;
   random  intercept /subject =roi_label_cha;
run;

proc glimmix data=cells;
	class roi_label_cha;
   model dfof = cell/ solution;
   random roi_label_cha / type=ar(1) residual;
run;

proc GENMOD data=cells;
   model dfof = cell_numeric sleep_numeric cell_numeric*sleep_numeric;
run;

