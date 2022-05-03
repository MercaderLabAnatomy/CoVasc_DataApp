pathsave="E:/Data_Drug_Screening_CovidBox/Example images/2x_examples/Experiment_ID_22/";
name=getTitle();
name=replace(name, "_1", "");
compound=getString("Compound name", "default");
name_rep = replace(name, "MAX", compound+"_CH_");
name_rep_C1 = replace(name_rep, "_CH_", "_C1");
name_rep_C2 = replace(name_rep, "_CH_", "_C2");
roiManager("Select", 0);
waitForUser("Place the rectangle");

run("Duplicate...", "duplicate title="+name+"");
run("Split Channels");

selectWindow("C1-"+name);
//setMinAndMax(6627, 20594);
setMinAndMax(2036, 20594);
print(name_rep)
wait(100);
saveAs("Jpeg", pathsave+"/" + name_rep_C1 +".jpg");

selectWindow("C2-"+name);

//setMinAndMax(456, 1784);
setMinAndMax(456, 1000);
run("Green");
wait(100);
saveAs("Jpeg", pathsave+"/" + name_rep_C2 +".jpg");

//run("Brightness/Contrast...");
//selectWindow("C1-MAX_H003_1-1");

//close();
//close();
